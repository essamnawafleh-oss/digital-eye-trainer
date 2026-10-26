"""Command line: python -m eye_trainer <command> [options]

Most people never need this: the start scripts open the local web page, which runs
the same steps with buttons. Commands:

  check            facts about this computer (Python, GPU, RAM, disk, models)
  download-models  download the three base models (the ONLY step that uses the internet)
  synth            generate synthetic training/validation/test lines
  import FOLDER    import PDFs/images: detect lines, crop, pre-fill suggestions
  ui               open the labelling page (http://127.0.0.1:8765)
  split            show/assign the train/validation/test split (by document)
  train            fine-tune (resumes automatically if interrupted)
  evaluate         before/after report on the test set(s)
  export           write the model folder for Digital Eye (only if the evaluation passed)
  verify-export    check an exported folder against its checksums and load it in PaddleOCR
  demo             synthetic-only end-to-end run (tests the installation)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .messages import configure_console, say
from .workspace import Workspace


def _ws(args) -> Workspace:
    return Workspace.default(args.workspace)


def cmd_check(args) -> int:
    from .pipeline import check_environment

    info = check_environment(_ws(args))
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if info["problems"]:
        print(say("تنبيهات:", "Problems:"))
        for p in info["problems"]:
            print(" -", p)
    return 0


def cmd_download(args) -> int:
    from .pipeline import download_models

    res = download_models(_ws(args))
    for name, r in res.items():
        print(name, "OK", "(Apache-2.0)" if r["license_apache_2_0"] else "(licence not found in model card!)")
    return 0


def cmd_synth(args) -> int:
    from .pipeline import SYNTH_DEFAULT, make_synthetic

    sizes = {"train": (args.train, SYNTH_DEFAULT["train"][1] + args.seed),
             "val": (args.val, SYNTH_DEFAULT["val"][1] + args.seed),
             "test": (args.test, SYNTH_DEFAULT["test"][1] + args.seed)}
    make_synthetic(_ws(args), sizes)
    return 0


def cmd_import(args) -> int:
    from .prep import Importer
    from .store import Store

    ws = _ws(args)
    store = Store(ws.root)
    try:
        print(json.dumps(Importer(ws, device=args.device).import_folder(Path(args.folder), store), indent=2))
        print(json.dumps(store.progress(), indent=2))
    finally:
        store.close()
    return 0


def cmd_split(args) -> int:
    from .store import Store

    store = Store(_ws(args).root)
    try:
        print(json.dumps(store.assign_splits(), indent=2))
    finally:
        store.close()
    return 0


def cmd_ui(args) -> int:
    from .ui.server import serve

    serve(_ws(args), port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_train(args) -> int:
    from .pipeline import run_training

    run_training(_ws(args), args.run, steps=args.steps, batch_size=args.batch_size, device=args.device,
                 use_real=not args.no_real, use_synthetic=not args.no_synthetic, real_fraction=args.real_fraction,
                 learning_rate=args.lr)
    return 0


def cmd_evaluate(args) -> int:
    from .pipeline import run_evaluation

    results = run_evaluation(_ws(args), args.run)
    for kind, rep in results.items():
        print(f"report: {_ws(args).report_dir(args.run) / f'evaluation_{kind}.md'}")
    return 0 if all(r["verdict"]["passed"] for r in results.values()) else 3


def cmd_export(args) -> int:
    from .pipeline import run_export

    run_export(_ws(args), args.run, args.name, force=args.force_without_passing_evaluation)
    return 0


def cmd_verify_export(args) -> int:
    from .export import check_loads_in_paddleocr, verify_files
    from .offline import offline

    problems = verify_files(args.folder)
    if problems:
        print(say("فشل التحقق:", "verification FAILED:"), *problems, sep="\n - ")
        return 1
    with offline():
        from .synth import load_fonts, render_line
        import random
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        img = tmp / "line.png"
        render_line("رقم القطعة: ١٢٥", load_fonts()[0], 36, random.Random(0)).save(img)
        check_loads_in_paddleocr(args.folder, [img])
    print(say("سليم: الملفات مطابقة للبصمات والنموذج يعمل في PaddleOCR.",
              "OK: files match their checksums and the model loads in PaddleOCR."))
    return 0


def cmd_demo(args) -> int:
    """Synthetic-only vertical slice: synth -> train -> evaluate -> export."""
    from .pipeline import make_synthetic, run_evaluation, run_export, run_training

    ws = _ws(args)
    t = time.time()
    make_synthetic(ws, {"train": (args.lines, 11), "val": (max(50, args.lines // 10), 22),
                        "test": (max(100, args.lines // 5), 33)})
    run = args.run or f"demo_{time.strftime('%Y%m%d_%H%M%S')}"
    run_training(ws, run, steps=args.steps, batch_size=args.batch_size, device=args.device, use_real=False)
    results = run_evaluation(ws, run)
    if all(r["verdict"]["passed"] for r in results.values()):
        run_export(ws, run)
    print(f"demo finished in {time.time() - t:.0f}s; reports in {ws.report_dir(run)}")
    return 0


def main(argv=None) -> int:
    configure_console()
    ap = argparse.ArgumentParser(prog="eye_trainer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", help="workspace folder (default ./workspace or $EYE_TRAINER_WORKSPACE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("download-models").set_defaults(fn=cmd_download)
    p = sub.add_parser("synth")
    p.add_argument("--train", type=int, default=6000)
    p.add_argument("--val", type=int, default=300)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("import")
    p.add_argument("folder")
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_import)
    sub.add_parser("split").set_defaults(fn=cmd_split)
    p = sub.add_parser("ui")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_ui)
    p = sub.add_parser("train")
    p.add_argument("--run", default="run1")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--real-fraction", type=float, default=0.5)
    p.add_argument("--no-real", action="store_true")
    p.add_argument("--no-synthetic", action="store_true")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", default="run1")
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("export")
    p.add_argument("--run", default="run1")
    p.add_argument("--name")
    p.add_argument("--force-without-passing-evaluation", action="store_true")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("verify-export")
    p.add_argument("folder")
    p.set_defaults(fn=cmd_verify_export)
    p = sub.add_parser("demo")
    p.add_argument("--run")
    p.add_argument("--lines", type=int, default=2000)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.set_defaults(fn=cmd_demo)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        if os.environ.get("EYE_TRAINER_DEBUG"):
            import traceback

            traceback.print_exc()
        print(f"\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
