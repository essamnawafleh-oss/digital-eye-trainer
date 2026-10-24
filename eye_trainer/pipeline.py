"""High-level steps shared by the command line and the local web page."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import synth
from .data import read_dataset
from .messages import say
from .offline import offline
from .workspace import ALL_MODELS, BASE_REC, Workspace

SYNTH_DEFAULT = {"train": (6000, 1101), "val": (300, 2202), "test": (1000, 3303)}  # lines, seed


# --------------------------------------------------------------------- environment
def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 2**30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except Exception:
        return None


def _version(pkg: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def check_environment(ws: Workspace) -> dict:
    """Facts about this machine and installation (no document data)."""
    info: dict = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "disk_free_gb": round(shutil.disk_usage(ws.root).free / 2**30, 1),
        "packages": {p: _version(p) for p in ("paddlepaddle", "paddlepaddle-gpu", "paddleocr", "paddlex",
                                             "pillow", "pypdfium2", "fonttools", "python-bidi", "numpy")},
        "pillow_raqm": synth.raqm_available(),
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
    }
    try:
        import paddle

        info["paddle_cuda_build"] = bool(paddle.device.is_compiled_with_cuda())
        info["gpu_count"] = paddle.device.cuda.device_count() if info["paddle_cuda_build"] else 0
    except Exception as exc:  # paddle missing or broken
        info["paddle_error"] = f"{type(exc).__name__}: {exc}"
    if info["nvidia_smi"]:
        try:
            info["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                           "--format=csv,noheader"], capture_output=True, text=True,
                                          timeout=20).stdout.strip().splitlines()
        except Exception:
            info["gpus"] = []
    info["models"] = {m: (ws.model_dir(m) / "inference.pdiparams").is_file() for m in ALL_MODELS}
    info["fonts"] = len(synth.load_fonts([ws.root / "fonts"]))
    info["workspace"] = str(ws.root)
    gpu = info.get("gpu_count", 0) > 0
    info["training_device"] = "gpu" if gpu else "cpu"
    problems = []
    if not info["pillow_raqm"]:
        problems.append("Pillow has no libraqm: Arabic synthetic lines cannot be rendered correctly")
    if not all(info["models"].values()):
        problems.append("base models missing: run download-models (needs internet once)")
    if info["ram_gb"] is not None and info["ram_gb"] < 4:
        problems.append("less than 4 GB RAM: use --batch-size 4")
    if info["disk_free_gb"] < 2:
        problems.append("less than 2 GB free disk")
    if info["nvidia_smi"] and not gpu:
        problems.append("an NVIDIA GPU is present but this Paddle build has no CUDA: training will use the CPU "
                        "(see README, GPU section)")
    info["problems"] = problems
    return info


# --------------------------------------------------------------------- models
def download_models(ws: Workspace, log=print) -> dict:
    """The only step that uses the internet. Downloads from Hugging Face (Apache-2.0)."""
    import logging

    from huggingface_hub import snapshot_download

    from .export import sha256

    for name in ("httpx", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)

    out = {}
    for name in ALL_MODELS:
        target = ws.model_dir(name)
        log(say(f"تنزيل {name} ...", f"downloading PaddlePaddle/{name} ..."))
        snapshot_download(repo_id=f"PaddlePaddle/{name}", local_dir=str(target),
                          allow_patterns=["inference.*", "README.md", "config.json"])
        files = {p.name: sha256(p) for p in sorted(target.iterdir()) if p.is_file()}
        if not {"inference.yml", "inference.json", "inference.pdiparams"} <= set(files):
            raise RuntimeError(f"{name}: download incomplete, run again")
        card = (target / "README.md").read_text(encoding="utf-8", errors="replace") if (target / "README.md").exists() else ""
        out[name] = {"files_sha256": files, "license_apache_2_0": "license: apache-2.0" in card.lower()}
    (ws.root / "models" / "manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------- data
def make_synthetic(ws: Workspace, sizes: dict | None = None, log=print) -> dict:
    sizes = sizes or SYNTH_DEFAULT
    fonts = synth.load_fonts([ws.root / "fonts"])
    log(say(f"عدد الخطوط المستخدمة: {len(fonts)}", f"fonts: {len(fonts)}"))
    out = {}
    with offline():
        for split, (n, seed) in sizes.items():
            folder = ws.synthetic(split)
            if folder.exists():
                shutil.rmtree(folder)
            synth.write_dataset(folder, n, seed, fonts, doc_prefix=f"synth{split}")
            out[split] = n
            log(say(f"أسطر اصطناعية ({split}): {n}", f"synthetic {split}: {n} lines"))
    return out


def prepare_real(ws: Workspace) -> dict:
    """Write labelled real lines to datasets/real_<split> and check the split has no leak."""
    from .store import Store

    store = Store(ws.root)
    try:
        sets = store.export_datasets(ws.root / "datasets")
        store.check_no_leak(sets)
        return {k: len(read_dataset(v)) for k, v in sets.items()}
    finally:
        store.close()


def _nonempty(folder: Path) -> bool:
    return (folder / "labels.jsonl").is_file() and (folder / "labels.jsonl").stat().st_size > 0


# --------------------------------------------------------------------- train
def run_training(ws: Workspace, run_name: str, steps: int = 3000, batch_size: int = 16, device: str = "auto",
                 use_real: bool = True, use_synthetic: bool = True, real_fraction: float = 0.5,
                 learning_rate: float = 1e-4, log=print, stop_flag=None) -> dict:
    from .train import TrainConfig, train

    real_counts = prepare_real(ws) if use_real else {}
    real_sets = [str(ws.real("train"))] if use_real and real_counts.get("train") else []
    syn_sets = [str(ws.synthetic("train"))] if use_synthetic and _nonempty(ws.synthetic("train")) else []
    val_sets = [str(p) for p in (ws.real("val"),) if use_real and real_counts.get("val")]
    if use_synthetic and _nonempty(ws.synthetic("val")):
        val_sets.append(str(ws.synthetic("val")))
    if not real_sets and not syn_sets:
        raise RuntimeError(say("لا توجد بيانات تدريب: صحّح بعض الأسطر أو أنشئ بيانات اصطناعية.",
                               "No training data: label some lines or generate synthetic data."))
    cfg = TrainConfig(base_model_dir=str(ws.base_rec), out_dir=str(ws.run_dir(run_name)), real_sets=real_sets,
                      synthetic_sets=syn_sets, val_sets=val_sets, real_fraction=real_fraction, max_steps=steps,
                      batch_size=batch_size, learning_rate=learning_rate, device=device)
    with offline():
        return train(cfg, log=log, stop_flag=stop_flag)


# --------------------------------------------------------------------- evaluate
def test_sets(ws: Workspace) -> list[tuple[str, Path]]:
    out = []
    if _nonempty(ws.real("test")):
        out.append(("real", ws.real("test")))
    if _nonempty(ws.synthetic("test")):
        out.append(("synthetic", ws.synthetic("test")))
    return out


def run_evaluation(ws: Workspace, run_name: str, log=print) -> dict:
    """Before/after on every available test set; writes shareable reports (numbers only)."""
    from .data import Charset, prepare
    from .evaluate import build_report, evaluate_records, report_markdown, write_private_errors
    from .model import load_from_inference
    from .train import load_best

    prepare_real(ws)
    sets = test_sets(ws)
    if not sets:
        raise RuntimeError(say("لا توجد مجموعة اختبار.", "No test set."))
    run_dir = ws.run_dir(run_name)
    train_docs = set()
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    for s in cfg["real_sets"] + cfg["synthetic_sets"]:
        train_docs |= {r.doc_id for r in read_dataset(s)}
    results = {}
    with offline():
        base, classes = load_from_inference(ws.base_rec)
        new, _ = load_best(run_dir, ws.base_rec)
        charset = Charset(classes)
        for kind, folder in sets:
            records = [p.record for p in prepare(read_dataset(folder), charset)[0]]
            leaked = {r.doc_id for r in records} & train_docs
            if leaked:
                raise RuntimeError(f"{len(leaked)} test documents were also used for training: evaluation refused")
            t0 = time.time()
            before, before_txt = evaluate_records(base, charset, records)
            after, after_txt = evaluate_records(new, charset, records)
            info = {"test_set": kind, "documents": len({r.doc_id for r in records}), "lines": len(records),
                    "documents_shared_with_training": 0, "run": run_name,
                    "training_steps": json.loads((run_dir / "checkpoint" / "state.json").read_text())["step"],
                    "seconds": round(time.time() - t0, 1)}
            if kind == "synthetic":
                info["note"] = ("synthetic lines from the same generator and fonts as the synthetic training data "
                                "(different seed and documents): this measures the pipeline, not real-world accuracy")
            rep = build_report(before, after, info)
            out_dir = ws.report_dir(run_name)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"evaluation_{kind}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / f"evaluation_{kind}.md").write_text(report_markdown(rep), encoding="utf-8")
            write_private_errors(ws.root / "private" / f"{run_name}_{kind}_errors.jsonl", records, before_txt, after_txt)
            results[kind] = rep
            v = rep["verdict"]
            log(say(("✓ نجح" if v["passed"] else "✗ فشل: النموذج الجديد أسوأ من الأصلي") + f" ({kind})",
                    ("PASSED" if v["passed"] else "FAILED: the new model is worse than the original") + f" ({kind})"))
    return results


def gating_report(ws: Workspace, run_name: str) -> Path | None:
    """Real test set if there is one, otherwise the synthetic one."""
    d = ws.report_dir(run_name)
    for kind in ("real", "synthetic"):
        if (d / f"evaluation_{kind}.json").is_file():
            return d / f"evaluation_{kind}.json"
    return None


def run_export(ws: Workspace, run_name: str, export_name: str | None = None, force: bool = False, log=print) -> Path:
    from .data import read_dataset as rd
    from .export import check_loads_in_paddleocr, export_run
    from .data import Charset
    from .evaluate import predict
    from .train import load_best

    reports = [p for p in (ws.report_dir(run_name) / "evaluation_real.json",
                           ws.report_dir(run_name) / "evaluation_synthetic.json") if p.is_file()]
    failed = [p for p in reports if not json.loads(p.read_text(encoding="utf-8"))["verdict"]["passed"]]
    if failed and not force:
        raise RuntimeError(say("لا يمكن التصدير: النموذج أسوأ من الأصلي في أحد الاختبارات.",
                               f"Refusing to export: failed evaluation in {', '.join(p.name for p in failed)}"))
    export_name = export_name or f"{BASE_REC}_ft_{run_name}"
    out = ws.root / "exports" / export_name
    with offline():
        model, classes = load_best(ws.run_dir(run_name), ws.base_rec)
        state = json.loads((ws.run_dir(run_name) / "checkpoint" / "state.json").read_text(encoding="utf-8"))
        info = {"run": run_name, "steps": state["step"], "best_val_cer": state["best_val_cer"],
                "base_val_cer": state["base_val_cer"]}
        export_run(model, ws.base_rec, out, info, gating_report(ws, run_name), force_unevaluated=force)
        # Load the exported folder with PaddleOCR itself and compare with the trained model.
        sample = []
        for kind, folder in test_sets(ws):
            sample += rd(folder)[:20]
        if sample:
            from .textutil import digital_eye_view

            expected = [digital_eye_view(t) for t, _ in predict(model, Charset(classes), sample, batch_size=1)]
            check_loads_in_paddleocr(out, [r.image for r in sample], expected)
            log(say(f"تم التحقق: النموذج المصدَّر يعمل داخل PaddleOCR ويعطي نفس النتائج ({len(sample)} سطر).",
                    f"verified: the exported model loads in PaddleOCR and gives identical text on {len(sample)} lines"))
    log(say(f"المجلد الجاهز: {out}", f"exported to {out}"))
    return out
