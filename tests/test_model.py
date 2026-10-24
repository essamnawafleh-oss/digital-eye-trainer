"""The trainable model reproduces the official inference model and exports back to it."""

import random

import numpy as np
import paddle
import pytest

from eye_trainer import data as D
from eye_trainer import synth
from eye_trainer.evaluate import predict
from eye_trainer.export import check_loads_in_paddleocr
from eye_trainer.model import WeightLoadError, export_inference, load_from_inference
from eye_trainer.textutil import digital_eye_view

from .conftest import requires_models

pytestmark = requires_models


@pytest.fixture(scope="module")
def arabic(base_models):
    paddle.set_device("cpu")
    return load_from_inference(base_models / "arabic_PP-OCRv5_mobile_rec")


def test_charset_matches_paddlex(arabic):
    _, classes = arabic
    assert classes[0] == "blank" and classes[-1] == " " and len(classes) == 749
    assert {"٫", "٬", "ي", "ك"} <= set(classes)


def test_english_model_also_loads(base_models):
    model, classes = load_from_inference(base_models / "en_PP-OCRv5_mobile_rec")
    assert len(classes) == 438


def test_same_text_as_paddleocr(arabic, base_models, fonts, tmp_path):
    model, classes = arabic
    folder = synth.write_dataset(tmp_path / "s", 25, 11, fonts)
    recs = D.read_dataset(folder)
    ours = [digital_eye_view(t) for t, _ in predict(model, D.Charset(classes), recs, batch_size=1)]
    theirs = check_loads_in_paddleocr(base_models / "arabic_PP-OCRv5_mobile_rec", [r.image for r in recs])
    assert ours == theirs


def test_export_round_trip_and_load_in_paddleocr(arabic, base_models, fonts, tmp_path):
    model, classes = arabic
    # change the weights a little so the export is really ours
    w = model.head.ctc_head.fc.bias
    w.set_value(w.numpy() + np.random.RandomState(0).normal(0, 0.05, w.shape).astype("float32"))
    out = export_inference(model, base_models / "arabic_PP-OCRv5_mobile_rec", tmp_path / "exp")
    assert {p.name for p in out.iterdir()} >= {"inference.json", "inference.pdiparams", "inference.yml"}
    reloaded, _ = load_from_inference(out)  # verifies outputs equal (max diff 1e-4)
    folder = synth.write_dataset(tmp_path / "s", 10, 12, fonts)
    recs = D.read_dataset(folder)
    expected = [digital_eye_view(t) for t, _ in predict(model, D.Charset(classes), recs, batch_size=1)]
    check_loads_in_paddleocr(out, [r.image for r in recs], expected)


def test_refuses_unknown_model(tmp_path, base_models):
    import shutil

    d = tmp_path / "m"
    shutil.copytree(base_models / "arabic_PP-OCRv5_mobile_rec", d, ignore=shutil.ignore_patterns(".cache"))
    yml = (d / "inference.yml").read_text(encoding="utf-8").replace("arabic_PP-OCRv5_mobile_rec", "other_rec")
    (d / "inference.yml").write_text(yml, encoding="utf-8")
    with pytest.raises(WeightLoadError):
        load_from_inference(d)


def test_render_seed_independent_of_global_random(fonts):
    a = synth.render_line("رقم ١٢", fonts[0], 30, random.Random(5))
    random.seed(99)
    b = synth.render_line("رقم ١٢", fonts[0], 30, random.Random(5))
    assert np.array_equal(np.asarray(a), np.asarray(b))
