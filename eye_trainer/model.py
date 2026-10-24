"""The PP-OCRv5 mobile text recogniser as a trainable Paddle model.

Why this file exists
--------------------
The official *training* checkpoints (``*_pretrained.pdparams``) are hosted on Baidu's
servers, while Digital Eye uses the *inference* export from Hugging Face. An inference
export is "re-parameterised": PP-LCNetV3's multi-branch training blocks are fused into
single convolutions and the NRTR training head is dropped. This module rebuilds the
recogniser in that fused form and loads the inference weights into it, so it can be
fine-tuned directly from the files Digital Eye already uses.

The parameter order inside ``inference.pdiparams`` carries no names. It is recovered by
exporting a copy of our model whose every tensor holds a unique constant, then reading
where each constant lands. The result is checked by comparing our model's output with
the official inference program on a fixed input; loading fails loudly on any mismatch.

Architecture: ``configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml``
(PaddleOCR v3.7.0). ``en_PP-OCRv5_mobile_rec`` uses the identical architecture.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import yaml

from .vendor.ppocr_rec.rec_ctc_head import CTCHead
from .vendor.ppocr_rec.rec_lcnetv3 import PPLCNetV3
from .vendor.ppocr_rec.rnn import Im2Seq, SequenceEncoder

INPUT_HEIGHT = 48
MIN_INPUT_WIDTH = 320
MODEL_FILES = ("inference.yml", "inference.json", "inference.pdiparams")


class WeightLoadError(RuntimeError):
    pass


@dataclass
class RecSpec:
    backbone_scale: float = 0.95
    neck: dict = field(default_factory=lambda: dict(
        encoder_type="svtr", dims=120, depth=2, hidden_dims=120, kernel_size=[1, 3], use_guide=True))
    head: dict = field(default_factory=lambda: dict(fc_decay=0.00001))


SUPPORTED_MODELS = {
    "arabic_PP-OCRv5_mobile_rec": RecSpec(),
    "en_PP-OCRv5_mobile_rec": RecSpec(),
}


class _Backbone(PPLCNetV3):
    """PP-LCNetV3 that always uses the inference-time pooling.

    Upstream training pools the feature map to a fixed 40 time steps
    (``adaptive_avg_pool2d``) and inference uses ``avg_pool2d([3, 2])``, which gives
    width/8 steps. For a 48x320 input both are identical; for wider lines only the
    inference pooling keeps one step per 8 pixels, so we train with it.
    """

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        return F.avg_pool2d(x, [3, 2])


class _CTCBranch(nn.Layer):
    """The CTC half of PaddleOCR's MultiHead, with the same attribute names."""

    def __init__(self, in_channels: int, num_classes: int, spec: RecSpec):
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        neck = dict(spec.neck)
        self.ctc_encoder = SequenceEncoder(in_channels=in_channels, encoder_type=neck.pop("encoder_type"), **neck)
        self.ctc_head = CTCHead(in_channels=self.ctc_encoder.out_channels, out_channels=num_classes, **spec.head)

    def forward(self, x):
        return self.ctc_head(self.ctc_encoder(x))


class RecModel(nn.Layer):
    """Input: float32 [N, 3, 48, W], BGR, scaled to [-1, 1].
    Output: [N, W/8, num_classes] logits in training mode, softmax in eval mode."""

    def __init__(self, num_classes: int, spec: RecSpec | None = None):
        super().__init__()
        spec = spec or RecSpec()
        self.backbone = _Backbone(scale=spec.backbone_scale)
        self.head = _CTCBranch(self.backbone.out_channels, num_classes, spec)
        for layer in self.sublayers():
            if hasattr(layer, "rep") and not getattr(layer, "is_repped", False):
                layer.rep()

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------- charset -------
def read_charset(model_dir: str | Path) -> list[str]:
    """Class list exactly as paddlex CTCLabelDecode builds it: blank, dict, space."""
    meta = yaml.safe_load(Path(model_dir, "inference.yml").read_text(encoding="utf-8"))
    post = meta.get("PostProcess", {})
    if post.get("name") != "CTCLabelDecode":
        raise WeightLoadError(f"{model_dir}: PostProcess is {post.get('name')!r}, expected CTCLabelDecode")
    chars = list(post["character_dict"])
    return ["blank"] + chars + [" "]


def model_name_of(model_dir: str | Path) -> str:
    meta = yaml.safe_load(Path(model_dir, "inference.yml").read_text(encoding="utf-8"))
    return meta["Global"]["model_name"]


def check_model_dir(model_dir: str | Path) -> Path:
    p = Path(model_dir)
    missing = [f for f in MODEL_FILES if not (p / f).is_file()]
    if missing:
        raise WeightLoadError(f"model folder {p} is missing {', '.join(missing)}")
    return p


# ---------------------------------------------------------------- loading -------
def _ctc_keys(model: RecModel) -> list[str]:
    return list(model.state_dict().keys())


def _saved_order(num_classes: int, spec: RecSpec) -> list[str]:
    """State-dict keys in the order paddle.jit.save writes them for this model.

    The order follows Paddle's generated tensor names, which depend on how many layers
    were created before. A fresh name scope reproduces the names of a model built
    first in a new process, which is how the official models were exported."""
    with paddle.utils.unique_name.guard():
        probe = RecModel(num_classes, spec)
    probe.eval()
    sd = probe.state_dict()
    keys = list(sd.keys())
    for i, k in enumerate(keys):
        sd[k].set_value(np.full(sd[k].shape, float(i + 1), dtype=sd[k].numpy().dtype))
    tmp = tempfile.mkdtemp(prefix="det_probe_")
    try:
        _jit_save(probe, tmp)
        saved = paddle.jit.load(os.path.join(tmp, "inference")).state_dict()
        order = []
        for j in range(len(saved)):
            v = saved[f"param_{j}"].numpy()
            idx = int(round(float(v.flat[0]))) - 1
            if not np.all(v == v.flat[0]) or not 0 <= idx < len(keys):
                raise WeightLoadError("could not determine the parameter order of the export")
            order.append(keys[idx])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if sorted(order) != sorted(keys):
        raise WeightLoadError("parameter order probe did not cover every tensor")
    return order


def load_from_inference(model_dir: str | Path, verify: bool = True) -> tuple[RecModel, list[str]]:
    """Build a trainable model initialised from an inference export (official or ours)."""
    model_dir = check_model_dir(model_dir)
    name = model_name_of(model_dir)
    if name not in SUPPORTED_MODELS:
        raise WeightLoadError(f"model '{name}' is not supported (supported: {', '.join(SUPPORTED_MODELS)})")
    spec = SUPPORTED_MODELS[name]
    charset = read_charset(model_dir)
    model = RecModel(len(charset), spec)
    ref = paddle.jit.load(str(model_dir / "inference"))
    ref_sd = ref.state_dict()
    order = _saved_order(len(charset), spec)
    if len(ref_sd) != len(order):
        raise WeightLoadError(f"{model_dir}: {len(ref_sd)} tensors in file, model expects {len(order)}")
    sd = model.state_dict()
    for j, key in enumerate(order):
        value = ref_sd[f"param_{j}"].numpy()
        if tuple(sd[key].shape) != value.shape:
            raise WeightLoadError(f"{model_dir}: tensor {j} has shape {value.shape}, expected {tuple(sd[key].shape)} for {key}")
        sd[key].set_value(value)
    model.eval()
    if verify:
        x = paddle.to_tensor(np.random.RandomState(0).uniform(-1, 1, (2, 3, INPUT_HEIGHT, 400)).astype("float32"))
        ours = model(x).numpy()
        theirs = ref(x)
        theirs = (theirs[0] if isinstance(theirs, (list, tuple)) else theirs).numpy()
        diff = float(np.abs(ours - theirs).max())
        if diff > 1e-4:
            raise WeightLoadError(f"loaded model differs from {model_dir} (max diff {diff:.2e}); refusing to continue")
    return model, charset


# ---------------------------------------------------------------- export --------
def _jit_save(model: RecModel, out_dir: str | Path) -> None:
    was_training = model.training
    model.eval()
    static = paddle.jit.to_static(
        model, input_spec=[paddle.static.InputSpec([None, 3, INPUT_HEIGHT, -1], "float32", name="x")],
        full_graph=True)
    paddle.jit.save(static, os.path.join(str(out_dir), "inference"))
    if was_training:
        model.train()


def export_inference(model: RecModel, base_model_dir: str | Path, out_dir: str | Path) -> Path:
    """Write inference.json / inference.pdiparams / inference.yml (paddlex format).

    inference.yml is copied from the base model: same model_name, dictionary and
    preprocessing, so PaddleOCR / Digital Eye load it like the original."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _jit_save(model, out)
    shutil.copy2(Path(base_model_dir) / "inference.yml", out / "inference.yml")
    for extra in ("inference.pdmodel",):
        (out / extra).unlink(missing_ok=True)
    check_model_dir(out)
    return out


def freeze_batchnorm(model: nn.Layer) -> None:
    """Keep BatchNorm running statistics fixed (small fine-tuning batches)."""
    for layer in model.sublayers():
        if isinstance(layer, (nn.BatchNorm, nn.BatchNorm1D, nn.BatchNorm2D)):
            layer.eval()
