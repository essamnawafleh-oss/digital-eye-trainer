"""Local label store (SQLite file in the workspace) and the document-level split.

Everything stays in ``workspace/labels.sqlite`` and ``workspace/crops/``. Original file
names are kept only here (so you can find a document again); they never appear in
reports. Document IDs are the first 16 hex digits of the file's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from . import textutil as tu
from .data import Record, write_dataset

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id TEXT PRIMARY KEY,
    file_name TEXT,            -- private, never exported to reports
    kind TEXT,
    pages INTEGER,
    split TEXT,                -- train | val | test, fixed once assigned
    added REAL,
    grp TEXT                   -- documents of one group always share a split
);
CREATE TABLE IF NOT EXISTS lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES docs(doc_id),
    page INTEGER, idx INTEGER,
    crop TEXT NOT NULL,        -- path relative to the workspace
    poly TEXT,
    suggestion TEXT, score REAL,
    alt_text TEXT, alt_score REAL,
    priority REAL DEFAULT 0,
    status TEXT DEFAULT 'todo',  -- todo | done | skipped | flagged
    label TEXT, direction TEXT DEFAULT 'auto',
    changed INTEGER DEFAULT 0,   -- 1 if the label differs from the suggestion
    updated REAL
);
CREATE INDEX IF NOT EXISTS lines_status ON lines(status, priority);
CREATE INDEX IF NOT EXISTS lines_doc ON lines(doc_id);
"""

SPLITS = ("train", "val", "test")
STATUSES = ("todo", "done", "skipped", "flagged")


def file_doc_id(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def priority(suggestion: str, score: float, alt_text: str | None) -> float:
    """Higher = shown earlier. Uncertain lines and lines with digits come first."""
    p = 1.0 - float(score or 0.0)
    if any(ch.isdigit() for ch in suggestion or ""):
        p += 0.5
    if alt_text is not None and tu.numbers(alt_text) != tu.numbers(suggestion or ""):
        p += 0.5  # the two recognisers disagree about the numbers
    if tu.has_arabic(suggestion or "") and tu.has_latin(alt_text or ""):
        p += 0.2  # mixed-script line
    return round(p, 4)


class Store:
    def __init__(self, workspace: str | Path):
        self.root = Path(workspace)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.root / "labels.sqlite"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(docs)")}
        if "grp" not in cols:  # workspace created by an older version
            self.db.execute("ALTER TABLE docs ADD COLUMN grp TEXT")
        self.db.commit()

    # ---------------------------------------------------------------- docs
    def has_doc(self, doc_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM docs WHERE doc_id=?", (doc_id,)).fetchone() is not None

    def add_doc(self, doc_id: str, file_name: str, kind: str, pages: int, group: str | None = None) -> None:
        """`group`: documents that belong together (e.g. re-scans or separate pages of one
        permit, kept in one sub-folder) always get the same split."""
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO docs(doc_id, file_name, kind, pages, added, grp) VALUES (?,?,?,?,?,?)",
                            (doc_id, file_name, kind, pages, time.time(), group or doc_id))
            self.db.commit()

    def add_lines(self, rows: list[dict]) -> None:
        with self._lock:
            self.db.executemany(
                "INSERT INTO lines(doc_id, page, idx, crop, poly, suggestion, score, alt_text, alt_score, priority, label, updated)"
                " VALUES (:doc_id,:page,:idx,:crop,:poly,:suggestion,:score,:alt_text,:alt_score,:priority,:suggestion,:t)",
                [{**r, "t": time.time(), "poly": json.dumps(r.get("poly"))} for r in rows])
            self.db.commit()

    # ---------------------------------------------------------------- labelling
    def next_lines(self, limit: int = 1, status: str = "todo") -> list[dict]:
        rows = self.db.execute("SELECT * FROM lines WHERE status=? ORDER BY priority DESC, id LIMIT ?",
                               (status, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_line(self, line_id: int) -> dict | None:
        r = self.db.execute("SELECT * FROM lines WHERE id=?", (line_id,)).fetchone()
        return dict(r) if r else None

    def save_label(self, line_id: int, status: str, label: str | None = None, direction: str = "auto",
                   persian_to_arabic: bool = True) -> dict:
        if status not in STATUSES:
            raise ValueError(f"bad status {status!r}")
        if direction not in ("auto", "rtl", "ltr"):
            raise ValueError(f"bad direction {direction!r}")
        line = self.get_line(line_id)
        if line is None:
            raise KeyError(line_id)
        if status == "done":
            label = tu.normalize_label(label or "", persian_to_arabic)
            if not label:
                raise ValueError("empty label: use skip or flag instead")
        else:
            label = line["label"]
        changed = int(label != tu.normalize_label(line["suggestion"] or "", persian_to_arabic))
        with self._lock:
            self.db.execute("UPDATE lines SET status=?, label=?, direction=?, changed=?, updated=? WHERE id=?",
                            (status, label, direction, changed, time.time(), line_id))
            self.db.commit()
        return self.get_line(line_id)

    def progress(self) -> dict:
        counts = {s: 0 for s in STATUSES}
        for r in self.db.execute("SELECT status, COUNT(*) n FROM lines GROUP BY status"):
            counts[r["status"]] = r["n"]
        docs = self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        corrected = self.db.execute("SELECT COUNT(*) FROM lines WHERE status='done' AND changed=1").fetchone()[0]
        return {"documents": docs, "lines": sum(counts.values()), **counts, "corrected": corrected,
                "splits": self.split_counts()}

    # ---------------------------------------------------------------- split
    def assign_splits(self, fractions: dict[str, float] | None = None) -> dict:
        """Give every document without a split one, by DOCUMENT GROUP (never by line).

        Existing assignments never change, so the test set stays fixed as you add
        documents. A document whose group already has a split joins it. New groups go to
        the split furthest below its target share (1st group -> train, 2nd -> test,
        3rd -> validation, so an honest test set exists as early as possible)."""
        fractions = fractions or {"train": 0.7, "val": 0.15, "test": 0.15}
        with self._lock:
            docs = self.db.execute("SELECT doc_id, split, COALESCE(grp, doc_id) g FROM docs").fetchall()
            group_split: dict[str, str] = {}
            for d in docs:
                if d["split"]:
                    group_split.setdefault(d["g"], d["split"])
            counts = {s: 0 for s in SPLITS}
            for sp in group_split.values():
                counts[sp] += 1
            new_groups = sorted({d["g"] for d in docs if d["g"] not in group_split},
                                key=lambda x: hashlib.sha256(x.encode()).hexdigest())
            for g in new_groups:
                total = sum(counts.values()) + 1
                if counts["train"] and not counts["test"]:
                    pick = "test"
                elif counts["train"] and counts["test"] and not counts["val"]:
                    pick = "val"
                else:
                    deficit = {s: fractions[s] * total - counts[s] for s in ("test", "val", "train")}
                    pick = max(deficit, key=lambda s: (deficit[s], s == "test", s == "val"))
                counts[pick] += 1
                group_split[g] = pick
            for d in docs:
                if not d["split"]:
                    self.db.execute("UPDATE docs SET split=? WHERE doc_id=?", (group_split[d["g"]], d["doc_id"]))
            self.db.commit()
        return self.split_counts()

    def split_counts(self) -> dict:
        out = {s: {"documents": 0, "labelled_lines": 0} for s in SPLITS}
        for r in self.db.execute("SELECT split, COUNT(*) n FROM docs WHERE split IS NOT NULL GROUP BY split"):
            out[r["split"]]["documents"] = r["n"]
        for r in self.db.execute("SELECT d.split, COUNT(*) n FROM lines l JOIN docs d USING(doc_id) "
                                 "WHERE l.status='done' AND d.split IS NOT NULL GROUP BY d.split"):
            out[r["split"]]["labelled_lines"] = r["n"]
        return out

    def export_datasets(self, out_root: str | Path) -> dict[str, Path]:
        """Write labelled lines to <out_root>/real_<split>/labels.jsonl (absolute image paths)."""
        self.assign_splits()
        out = {}
        for split in SPLITS:
            rows = self.db.execute(
                "SELECT l.* FROM lines l JOIN docs d USING(doc_id) WHERE l.status='done' AND d.split=? ORDER BY l.id",
                (split,)).fetchall()
            recs = [Record((self.root / r["crop"]).resolve(), r["label"], r["doc_id"], "real", r["direction"] or "auto")
                    for r in rows]
            out[split] = write_dataset(Path(out_root) / f"real_{split}", recs)
        return out

    def check_no_leak(self, datasets: dict[str, Path]) -> None:
        """Fail if any document appears in more than one split."""
        from .data import read_dataset

        seen: dict[str, str] = {}
        for split, folder in datasets.items():
            for r in read_dataset(folder):
                if seen.setdefault(r.doc_id, split) != split:
                    raise RuntimeError(f"document {r.doc_id} appears in both {seen[r.doc_id]} and {split}")

    def close(self) -> None:
        self.db.close()
