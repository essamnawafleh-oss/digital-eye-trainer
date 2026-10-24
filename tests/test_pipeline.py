"""End to end on a tiny synthetic dataset: train a few steps, resume, evaluate, export,
refuse a worse model, keep reports free of document text."""

import json
import random

import numpy as np
import paddle
import pytest

from eye_trainer import pipeline, synth
from eye_trainer.evaluate import score, verdict
from eye_trainer.export import verify_files
from eye_trainer.prep import Importer
from eye_trainer.store import Store

from .conftest import requires_models

pytestmark = requires_models
QUIET = dict(log=lambda m: None)


@pytest.fixture()
def tiny(workspace):
    pipeline.make_synthetic(workspace, {"train": (32, 1), "val": (8, 2), "test": (16, 3)}, **QUIET)
    return workspace


def test_train_resume_evaluate_export(tiny):
    ws = tiny
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    st = pipeline.run_training(ws, "r", steps=4, batch_size=4, use_real=False, stop_flag=stop_after_two, **QUIET)
    assert st["status"] == "paused" and st["step"] == 2
    st = pipeline.run_training(ws, "r", steps=4, batch_size=4, use_real=False, **QUIET)
    assert st["status"] == "finished" and st["step"] == 4
    prog = json.loads((ws.run_dir("r") / "progress.json").read_text())
    assert prog["percent"] == 100.0 and prog["base_val_cer"] is not None

    res = pipeline.run_evaluation(ws, "r", **QUIET)
    rep = res["synthetic"]
    assert set(rep["before"]) == {"digital_eye", "raw_model"}
    assert (ws.report_dir("r") / "evaluation_synthetic.md").is_file()

    out = pipeline.run_export(ws, "r", force=not rep["verdict"]["passed"], **QUIET)
    assert verify_files(out) == []
    man = json.loads((out / "manifest.json").read_text())
    assert man["model_name"] == "arabic_PP-OCRv5_mobile_rec" and man["confidential"]
    assert "rec_model_dir" in (out / "DIGITAL_EYE_SETTINGS.txt").read_text(encoding="utf-8")
    (out / "inference.yml").write_text("tampered", encoding="utf-8")
    assert verify_files(out) == ["checksum mismatch: inference.yml"]


def test_changed_settings_refuse_to_resume(tiny):
    pipeline.run_training(tiny, "r", steps=1, batch_size=4, use_real=False, **QUIET)
    with pytest.raises(RuntimeError):
        pipeline.run_training(tiny, "r", steps=1, batch_size=8, use_real=False, **QUIET)


def test_worse_model_fails_loudly_and_is_not_exported(tiny):
    ws = tiny
    pipeline.run_training(ws, "bad", steps=1, batch_size=4, use_real=False, **QUIET)
    best = ws.run_dir("bad") / "checkpoint" / "best.pdparams"
    sd = paddle.load(str(best))
    rng = np.random.RandomState(0)
    sd = {k: (v.numpy() + rng.normal(0, 0.5, v.shape).astype(v.numpy().dtype)) if "head" in k else v
          for k, v in sd.items()}
    paddle.save(sd, str(best))
    res = pipeline.run_evaluation(ws, "bad", **QUIET)
    assert not res["synthetic"]["verdict"]["passed"]
    assert "FAILED" in (ws.report_dir("bad") / "evaluation_synthetic.md").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        pipeline.run_export(ws, "bad", **QUIET)


def test_reports_contain_no_document_text(tiny):
    ws = tiny
    pipeline.run_training(ws, "p", steps=1, batch_size=4, use_real=False, **QUIET)
    pipeline.run_evaluation(ws, "p", **QUIET)
    from eye_trainer.data import read_dataset

    texts = {r.text for r in read_dataset(ws.synthetic("test"))}
    words = {w for t in texts for w in t.split() if len(w) >= 4}
    for f in ws.report_dir("p").iterdir():
        content = f.read_text(encoding="utf-8")
        leaked = [w for w in words if w in content]
        assert not leaked, f"{f.name} contains document text: {leaked[:3]}"
    # the per-line error file exists, but only under private/
    assert any((ws.root / "private").iterdir())


def test_verdict_logic():
    gts = ["رقم ١٢٫٥", "Plot 128"]
    good = score(gts, gts)
    bad = score(gts, ["رقم ١٢٥", "Plot 18"])
    both = lambda s: {"digital_eye": s, "raw_model": s}  # noqa: E731
    assert verdict(both(bad), both(good))["passed"]
    v = verdict(both(good), both(bad))
    assert not v["passed"] and v["problems"]
    assert good["all"]["number_exact_match"] == 1.0 and bad["all"]["number_exact_match"] == 0.0


def test_import_documents(workspace, fonts):
    rng = random.Random(4)
    inbox = workspace.root / "inbox"
    synth.render_page([synth.make_line(rng) for _ in range(6)], fonts, rng).save(inbox / "page.png")
    p1 = synth.render_page([synth.make_line(rng) for _ in range(5)], fonts, rng)
    p1.save(inbox / "doc.pdf", save_all=True, append_images=[p1.copy()], resolution=200)
    store = Store(workspace.root)
    s = Importer(workspace).import_folder(inbox, store, **QUIET)
    assert s["imported"] == 2 and s["failed"] == 0 and s["lines"] >= 12
    again = Importer(workspace).import_folder(inbox, store, **QUIET)
    assert again["already_imported"] == 2 and again["lines"] == 0
    line = store.next_lines(1)[0]
    assert (workspace.root / line["crop"]).is_file() and line["suggestion"] is not None
    pages = store.db.execute("SELECT pages FROM docs ORDER BY pages").fetchall()
    assert [p[0] for p in pages] == [1, 2]
    assert store.progress()["splits"]["test"]["documents"] == 1
