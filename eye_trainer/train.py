"""Fine-tuning loop: CPU or NVIDIA GPU, resumable, with plain-language progress."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import paddle

from . import data as D
from .evaluate import evaluate_records
from .messages import say
from .model import freeze_batchnorm, load_from_inference


@dataclass
class TrainConfig:
    base_model_dir: str
    out_dir: str
    real_sets: list[str] = field(default_factory=list)       # labelled real lines (train split)
    synthetic_sets: list[str] = field(default_factory=list)  # generated lines
    val_sets: list[str] = field(default_factory=list)
    real_fraction: float = 0.5   # share of real lines in each epoch when both kinds exist
    max_steps: int = 3000
    batch_size: int = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    eval_every: int = 250
    save_every: int = 50
    max_val_lines: int = 600
    seed: int = 0
    device: str = "auto"  # auto | cpu | gpu
    augment: bool = True

    def fingerprint(self) -> str:
        """Settings that must not change when resuming."""
        keys = ("base_model_dir", "real_sets", "synthetic_sets", "real_fraction", "max_steps",
                "batch_size", "learning_rate", "warmup_steps", "seed")
        blob = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pick_device(wanted: str) -> str:
    gpu_ok = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    if wanted == "gpu" and not gpu_ok:
        raise RuntimeError(say("لا توجد بطاقة NVIDIA مع CUDA صالحة لهذا الإصدار من Paddle.",
                               "No usable NVIDIA GPU/CUDA for this Paddle build."))
    device = "gpu" if (wanted in ("auto", "gpu") and gpu_ok) else "cpu"
    paddle.set_device(device)
    return device


def mix(real: list, synthetic: list, real_fraction: float, rng: random.Random) -> list:
    """Repeat the smaller pool so that real lines make up ~real_fraction of an epoch."""
    if not real or not synthetic:
        return list(real or synthetic)
    want_real = real_fraction / max(1e-6, 1 - real_fraction) * len(synthetic)
    reps = max(1, round(want_real / len(real)))
    return real * reps + synthetic


def _human(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, m = divmod(seconds // 60, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {seconds % 60}s" if m else f"{seconds}s"


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _save(path_base: Path, model, opt=None) -> None:
    tmp = path_base.with_name(path_base.name + ".tmp")
    paddle.save(model.state_dict(), str(tmp) + ".pdparams")
    os.replace(str(tmp) + ".pdparams", str(path_base) + ".pdparams")
    if opt is not None:
        paddle.save(opt.state_dict(), str(tmp) + ".pdopt")
        os.replace(str(tmp) + ".pdopt", str(path_base) + ".pdopt")


def validation_cer(model, charset, records) -> float:
    """Model selection uses what Digital Eye will actually receive."""
    return evaluate_records(model, charset, records)[0]["digital_eye"]["all"]["cer"]


def train(cfg: TrainConfig, log=print, stop_flag=None) -> dict:
    """Run (or resume) a fine-tuning run. Returns the final state dict.

    Paddle names parameters and optimizer moments with global counters; a fresh name
    scope makes them identical in every session, which resuming from a checkpoint needs."""
    with paddle.utils.unique_name.guard():
        return _train(cfg, log, stop_flag)


def _train(cfg: TrainConfig, log, stop_flag) -> dict:
    out = Path(cfg.out_dir)
    ck = out / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    paddle.seed(cfg.seed)

    model, classes = load_from_inference(cfg.base_model_dir)
    charset = D.Charset(classes)

    real = [r for s in cfg.real_sets for r in D.read_dataset(s)]
    synthetic = [r for s in cfg.synthetic_sets for r in D.read_dataset(s)]
    real_p, real_skip = D.prepare(real, charset)
    syn_p, syn_skip = D.prepare(synthetic, charset)
    items = mix(real_p, syn_p, cfg.real_fraction, rng)
    if not items:
        raise RuntimeError(say("لا توجد أسطر صالحة للتدريب.", "No usable training lines."))
    val = [r for s in cfg.val_sets for r in D.read_dataset(s)]
    val = [p.record for p in D.prepare(val, charset)[0]]
    if len(val) > cfg.max_val_lines:
        val = random.Random(1).sample(val, cfg.max_val_lines)

    sched = paddle.optimizer.lr.LinearWarmup(
        paddle.optimizer.lr.CosineAnnealingDecay(cfg.learning_rate, T_max=max(1, cfg.max_steps - cfg.warmup_steps)),
        warmup_steps=cfg.warmup_steps, start_lr=0.0, end_lr=cfg.learning_rate)
    opt = paddle.optimizer.Adam(learning_rate=sched, parameters=model.parameters(),
                                grad_clip=paddle.nn.ClipGradByGlobalNorm(5.0))
    ctc = paddle.nn.CTCLoss(blank=0, reduction="mean")

    state_file = ck / "state.json"
    state = {"step": 0, "fingerprint": cfg.fingerprint(), "best_val_cer": None, "base_val_cer": None,
             "history": [], "status": "running", "device": device, "seconds": 0.0}
    if state_file.exists():
        old = json.loads(state_file.read_text(encoding="utf-8"))
        if old.get("fingerprint") != state["fingerprint"]:
            raise RuntimeError(say(
                "يوجد تدريب سابق في هذا المجلد بإعدادات مختلفة. اختر مجلداً جديداً أو احذف القديم.",
                f"{out} holds a run with different settings. Use a new output folder or delete the old one."))
        if (ck / "last.pdparams").exists():
            model.set_state_dict(paddle.load(str(ck / "last.pdparams")))
            opt.set_state_dict(paddle.load(str(ck / "last.pdopt")))
            state.update({k: old[k] for k in ("step", "best_val_cer", "base_val_cer", "history", "seconds") if k in old})
            rng.seed(cfg.seed + state["step"])
            log(say(f"استئناف التدريب من الخطوة {state['step']}.", f"Resuming from step {state['step']}."))
    _atomic_json(out / "config.json", asdict(cfg))
    _atomic_json(out / "data_summary.json", {
        "real_lines_used": len(real_p), "synthetic_lines_used": len(syn_p), "validation_lines": len(val),
        "real_lines_skipped": real_skip, "synthetic_lines_skipped": syn_skip, "device": device})
    log(say(f"الجهاز: {'بطاقة رسومات' if device == 'gpu' else 'المعالج (CPU)'} — أسطر حقيقية: {len(real_p)}، "
            f"اصطناعية: {len(syn_p)}، تحقق: {len(val)}",
            f"device: {device}; real lines: {len(real_p)}, synthetic: {len(syn_p)}, validation: {len(val)}"))

    if val and state["base_val_cer"] is None:
        state["base_val_cer"] = validation_cer(model, charset, val)
        if state["best_val_cer"] is None:
            state["best_val_cer"] = state["base_val_cer"]
            _save(ck / "best", model)
        log(say(f"نسبة خطأ الأحرف للنموذج الأصلي على بيانات التحقق: {state['base_val_cer']:.2%}",
                f"original model validation CER: {state['base_val_cer']:.2%}"))
    if not val and not (ck / "best.pdparams").exists():
        _save(ck / "best", model)

    def progress(status: str, loss=None, eta=None):
        state["status"] = status
        _atomic_json(state_file, state)
        _atomic_json(out / "progress.json", {
            "status": status, "step": state["step"], "max_steps": cfg.max_steps,
            "percent": round(100 * state["step"] / cfg.max_steps, 1), "eta_seconds": eta, "loss": loss,
            "base_val_cer": state["base_val_cer"], "best_val_cer": state["best_val_cer"], "device": device})

    stream = D.batches(items, cfg.batch_size, rng)
    model.train()
    freeze_batchnorm(model)
    step_times: list[float] = []
    loss_avg = None
    progress("running")
    try:
        while state["step"] < cfg.max_steps:
            if stop_flag is not None and stop_flag():
                raise KeyboardInterrupt
            t0 = time.time()
            batch = D.make_train_batch(next(stream), rng, cfg.augment)
            x, y, lengths, steps = batch
            if len(y) == 0:
                continue
            logits = model(paddle.to_tensor(x))  # [N, T, C]
            loss = ctc(logits.transpose([1, 0, 2]), paddle.to_tensor(y),
                       paddle.to_tensor(np.full(len(y), steps, dtype="int64")), paddle.to_tensor(lengths))
            loss.backward()
            opt.step()
            opt.clear_grad()
            sched.step()
            state["step"] += 1
            lv = float(loss)
            loss_avg = lv if loss_avg is None else 0.95 * loss_avg + 0.05 * lv
            dt = time.time() - t0
            state["seconds"] += dt
            step_times = (step_times + [dt])[-50:]
            eta = (cfg.max_steps - state["step"]) * (sum(step_times) / len(step_times))
            if state["step"] % 10 == 0 or state["step"] == 1:
                log(say(f"الخطوة {state['step']} من {cfg.max_steps} ({100 * state['step'] / cfg.max_steps:.0f}%) — "
                        f"الوقت المتبقي تقريباً {_human(eta)} — مقدار الخطأ {loss_avg:.3f}",
                        f"step {state['step']}/{cfg.max_steps}, ~{_human(eta)} left, loss {loss_avg:.3f}"))
            if state["step"] % cfg.save_every == 0:
                _save(ck / "last", model, opt)
                progress("running", loss_avg, eta)
            if val and (state["step"] % cfg.eval_every == 0 or state["step"] == cfg.max_steps):
                v = validation_cer(model, charset, val)
                model.train()
                freeze_batchnorm(model)
                state["history"].append({"step": state["step"], "val_cer": v, "loss": loss_avg})
                better = v < state["best_val_cer"]
                if better:
                    state["best_val_cer"] = v
                    _save(ck / "best", model)
                log(say(f"تحقق: نسبة خطأ الأحرف {v:.2%} (الأصلي {state['base_val_cer']:.2%}، الأفضل {state['best_val_cer']:.2%})"
                        + (" ✓ حُفظ كأفضل نموذج" if better else ""),
                        f"validation CER {v:.2%} (original {state['base_val_cer']:.2%}, best {state['best_val_cer']:.2%})"))
                _save(ck / "last", model, opt)  # state.json must never be ahead of the saved weights
                progress("running", loss_avg, eta)
        if not val:
            _save(ck / "best", model)
        _save(ck / "last", model, opt)
        progress("finished", loss_avg, 0)
        log(say("انتهى التدريب.", "Training finished."))
    except KeyboardInterrupt:
        _save(ck / "last", model, opt)
        progress("paused", loss_avg, None)
        log(say("أُوقف التدريب وحُفظ التقدم. شغّل نفس الأمر لاحقاً للمتابعة.",
                "Training paused and saved. Run the same command again to resume."))
    return state


def load_best(run_dir: str | Path, base_model_dir: str | Path):
    """The best checkpoint of a run as an eval-mode model (+ class list)."""
    model, classes = load_from_inference(base_model_dir, verify=False)
    model.set_state_dict(paddle.load(str(Path(run_dir) / "checkpoint" / "best.pdparams")))
    model.eval()
    return model, classes

