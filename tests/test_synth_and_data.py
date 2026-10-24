import random

import numpy as np
import pytest

from eye_trainer import data as D
from eye_trainer import synth
from eye_trainer import textutil as tu


def test_bundled_fonts_cover_arabic_and_digits(fonts):
    assert len(fonts) >= 5
    for f in fonts:
        assert f.covers("رقم القطعة ١٢٣٫٤ 0123 ABC")


def test_raqm_available():
    assert synth.raqm_available(), "Pillow needs libraqm for correct Arabic rendering"


def test_generator_is_deterministic_and_grouped_by_document(fonts):
    a = [s.text for s in synth.generate(30, 7, fonts)]
    b = [s.text for s in synth.generate(30, 7, fonts)]
    assert a == b
    docs = [s.doc_id for s in synth.generate(30, 7, fonts, lines_per_doc=10)]
    assert len(set(docs)) == 3


def test_generator_covers_the_known_weaknesses(fonts):
    texts = [s.text for s in synth.generate(400, 3, fonts, strength=(0, 0))]
    cats = {c for t in texts for c in tu.categories(t)}
    assert {"arabic_decimal", "ascii_digits_in_arabic", "mixed_script", "ids", "arabic_indic_digits"} <= cats
    assert any("م2" in t or "دونم" in t for t in texts)
    assert not any(ch in t for t in texts for ch in tu.PERSIAN_TO_ARABIC)


def test_rendered_line_is_not_blank(fonts):
    img = synth.render_line("المساحة: ٢٨٩٣٫٦٦ م2", fonts[0], 36, random.Random(0))
    arr = np.asarray(img)
    assert arr.min() < 100 and arr.max() > 200


def test_degrade_keeps_size_reasonable(fonts):
    img = synth.render_line("Plot No: 128", fonts[0], 30, random.Random(1))
    out = synth.degrade(img, random.Random(2), 1.0)
    assert out.mode == "RGB" and out.width >= img.width * 0.9


def test_dataset_write_read(tmp_path, fonts):
    folder = synth.write_dataset(tmp_path / "s", 12, 5, fonts, lines_per_doc=4)
    recs = D.read_dataset(folder)
    assert len(recs) == 12 and all(r.image.is_file() for r in recs)
    assert len({r.doc_id for r in recs}) == 3


def test_charset_encode_and_ctc_decode():
    cs = D.Charset(["blank", "a", "b", " "])
    assert cs.encode("ab a") == [1, 2, 3, 1]
    assert cs.encode("abc") is None and cs.unsupported("abc") == {"c"}
    probs = np.eye(4)[[1, 1, 0, 1, 2, 2, 0]]
    assert cs.decode(probs)[0] == "aab"


def test_ctc_fits():
    assert D.ctc_fits(list(range(1, 31)), 320)
    assert not D.ctc_fits(list(range(1, 41)), 320)
    assert not D.ctc_fits([1, 1] * 13, 320)  # repeats need blanks between them


def test_batch_tensor_matches_paddlex_preprocessing():
    pytest.importorskip("paddlex")
    from paddlex.inference.models.text_recognition.processors import OCRReisizeNormImg

    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for h, w in ((30, 90), (40, 500), (25, 60))]
    ours = D.batch_tensor(imgs)
    op = OCRReisizeNormImg(rec_image_shape=[3, 48, 320])
    max_ratio = max(320 / 48, *[i.shape[1] / i.shape[0] for i in imgs])
    theirs = np.stack([op.resize_norm_img(i, max_ratio) for i in imgs])
    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs)


def test_prepare_skips_unsupported_and_reports_codepoints_only(tmp_path, fonts):
    folder = synth.write_dataset(tmp_path / "s", 3, 1, fonts)
    recs = D.read_dataset(folder)
    recs[0].text = "سطر ☃"
    cs = D.Charset(["blank"] + sorted({c for r in recs for c in r.target if c != "☃"}))
    kept, skipped = D.prepare(recs, cs)
    assert len(kept) == 2
    assert skipped["unsupported_characters"] == 1
    assert skipped["unsupported_codepoints"] == ["U+2603"]
