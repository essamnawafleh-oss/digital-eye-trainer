from pathlib import Path

import pytest
from PIL import Image

from eye_trainer.data import read_dataset
from eye_trainer.store import Store, priority


def _add_doc(store: Store, root: Path, doc: str, n: int = 3, group: str | None = None):
    (root / "crops" / doc).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rel = f"crops/{doc}/l{i}.png"
        Image.new("RGB", (100, 30), "white").save(root / rel)
        rows.append({"doc_id": doc, "page": 1, "idx": i, "crop": rel, "poly": [[0, 0]], "suggestion": f"رقم {i}",
                     "score": 0.9 - i / 10, "alt_text": None, "alt_score": None, "priority": i})
    store.add_doc(doc, f"secret_name_{doc}.pdf", "image", 1, group)
    store.add_lines(rows)


def test_label_lifecycle(empty_workspace):
    s = Store(empty_workspace.root)
    _add_doc(s, empty_workspace.root, "d1")
    first = s.next_lines(1)[0]
    assert first["idx"] == 2  # highest priority first
    s.save_label(first["id"], "done", "رقم  القطعة ١٢٫٥", "rtl")
    s.save_label(s.next_lines(1)[0]["id"], "skipped")
    s.save_label(s.next_lines(1)[0]["id"], "flagged")
    p = s.progress()
    assert (p["done"], p["skipped"], p["flagged"], p["todo"], p["corrected"]) == (1, 1, 1, 0, 1)
    assert s.get_line(first["id"])["label"] == "رقم القطعة ١٢٫٥"
    with pytest.raises(ValueError):
        s.save_label(first["id"], "done", "   ")
    with pytest.raises(ValueError):
        s.save_label(first["id"], "maybe", "x")


def test_persian_letters_converted_on_save(empty_workspace):
    s = Store(empty_workspace.root)
    _add_doc(s, empty_workspace.root, "d1", 1)
    line = s.next_lines(1)[0]
    assert s.save_label(line["id"], "done", "لیلى")["label"] == "ليلى"


def test_progress_survives_reopen(empty_workspace):
    s = Store(empty_workspace.root)
    _add_doc(s, empty_workspace.root, "d1", 2)
    s.save_label(s.next_lines(1)[0]["id"], "done", "نص")
    s.close()
    s2 = Store(empty_workspace.root)
    assert s2.progress()["done"] == 1


def test_split_is_by_document_stable_and_has_a_test_set(empty_workspace):
    s = Store(empty_workspace.root)
    for i in range(2):
        _add_doc(s, empty_workspace.root, f"doc{i}")
    first = s.assign_splits()
    assert first["test"]["documents"] == 1 and first["train"]["documents"] == 1
    before = {r["doc_id"]: r["split"] for r in s.db.execute("SELECT doc_id, split FROM docs")}
    for i in range(2, 20):
        _add_doc(s, empty_workspace.root, f"doc{i}")
    counts = s.assign_splits()
    after = {r["doc_id"]: r["split"] for r in s.db.execute("SELECT doc_id, split FROM docs")}
    assert all(after[d] == sp for d, sp in before.items()), "existing assignments must never change"
    assert counts["test"]["documents"] >= 2 and counts["val"]["documents"] >= 2
    assert sum(c["documents"] for c in counts.values()) == 20


def test_export_datasets_never_mixes_documents(empty_workspace):
    s = Store(empty_workspace.root)
    for i in range(6):
        _add_doc(s, empty_workspace.root, f"doc{i}")
    for line in s.next_lines(100):
        s.save_label(line["id"], "done", "نص " + str(line["id"]))
    sets = s.export_datasets(empty_workspace.root / "datasets")
    s.check_no_leak(sets)
    seen = {}
    for split, folder in sets.items():
        for r in read_dataset(folder):
            assert seen.setdefault(r.doc_id, split) == split
    assert sum(len(read_dataset(f)) for f in sets.values()) == 18


def test_priority_prefers_uncertain_lines_with_digits():
    assert priority("رقم 128", 0.95, "128") > priority("رقم", 0.95, None)
    assert priority("نص", 0.5, None) > priority("نص", 0.99, None)
    assert priority("رقم 18", 0.9, "Plot 128") > priority("رقم 18", 0.9, "Plot 18")


def test_documents_of_one_group_share_a_split(empty_workspace):
    s = Store(empty_workspace.root)
    for i in range(12):
        _add_doc(s, empty_workspace.root, f"doc{i}", 1, group=f"permit{i // 3}")  # 3 scans per permit
    s.assign_splits()
    _add_doc(s, empty_workspace.root, "late_rescan", 1, group="permit1")
    s.assign_splits()
    rows = s.db.execute("SELECT grp, split FROM docs").fetchall()
    by_group = {}
    for g, sp in rows:
        by_group.setdefault(g, set()).add(sp)
    assert all(len(v) == 1 for v in by_group.values())
    assert {next(iter(v)) for v in by_group.values()} == {"train", "test", "val"}
