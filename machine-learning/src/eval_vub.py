from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import transformer_suite as ts
from metrics.metrics_suite import MetricsSuite
from train_vub import (
    REGIMES,
    _make_serializable,
    build_test_loader,
    evaluate_regimes,
    make_hyperparameters,
    membrane_uuids,
    resolve_membrane_paths,
    resolve_paths,
    write_summary_csv,
)
from utils.hyperparameters_transformer import Hyperparameters
from utils.utils_transformer import normalize_device_string, setup_torch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eval-only for the V/U/B study.")
    p.add_argument("--study-dir", type=str, default="experiments/vub")
    p.add_argument("--mode", type=str, default=None,
                   choices=["voiced", "unvoiced", "both"])
    p.add_argument("--run-dir", type=str, default=None,
                   help="Direct path to a run folder (overrides --study-dir/--mode).")
    # data (must match training)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--voiced-path", type=str, default=None)
    p.add_argument("--unvoiced-path", type=str, default=None)
    p.add_argument("--textgrid-path", type=str, default=None)
    p.add_argument("--membrane-data-path", type=str, default=None)
    p.add_argument("--membrane-voiced-path", type=str, default=None)
    p.add_argument("--membrane-unvoiced-path", type=str, default=None)
    p.add_argument("--membrane-textgrid-path", type=str, default=None)
    p.add_argument("--membrane-uuids", type=str, default=None)
    # identity
    p.add_argument("--seed", type=int, default=3401)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cuda-device", type=int, default=None)
    # eval knobs
    p.add_argument("--vocode-samples", type=int, default=10)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--no-waveform-metrics", action="store_true")
    p.add_argument("--compute-efficiency", action="store_true")
    return p

def main() -> None:
    args = build_parser().parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        if args.mode is None:
            sys.exit("Provide --mode (with --study-dir) or --run-dir.")
        run_dir = Path(args.study_dir) / args.mode
    if not run_dir.exists():
        sys.exit(f"Run dir not found: {run_dir}")

    mode = args.mode or run_dir.name
    checkpoint = run_dir / "model_final.pt"
    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint} (train_vub.py should have written it)")

    # Device / torch setup.
    hp0 = Hyperparameters()
    if args.cuda_device is not None:
        hp0.device = f"cuda:{args.cuda_device}"
    hp0.device = normalize_device_string(hp0.device)
    hp0.device = setup_torch(hp0.device)

    # Rebuild hyperparameters
    hp = make_hyperparameters(args, hp0)

    voiced_path, unvoiced_path, textgrid_path = resolve_paths(args)
    test_loader = build_test_loader(hp, voiced_path, unvoiced_path, textgrid_path)
    print(f"[eval] Main test split: {len(test_loader.dataset)} utterances (seed={hp.seed})")

    model, _, _ = ts.init_model_and_optimizer(hp, checkpoint_path=str(checkpoint))

    # Main test set, under both input regimes.
    reports_main = evaluate_regimes(model, hp, test_loader, REGIMES, args, history=None)
    for regime, report in reports_main.items():
        with open(run_dir / f"metrics_{mode}_{regime}.json", "w") as f:
            json.dump(_make_serializable(report), f, indent=2)
        print(f"[eval] wrote {run_dir / f'metrics_{mode}_{regime}.json'}")

    # Membrane held-out set (optional).
    base, m_voiced, m_unvoiced, m_textgrid = resolve_membrane_paths(args)
    if base.exists() and any(p.exists() for p in (m_voiced, m_unvoiced, m_textgrid)):
        membrane_loader = ts.load_membrane_test_loader(
            hp, m_voiced, m_unvoiced, m_textgrid, membrane_uuids(args), debug=False,
        )
        reports_membrane = evaluate_regimes(model, hp, membrane_loader, REGIMES, args, history=None)
        for regime, report in reports_membrane.items():
            with open(run_dir / f"metrics_{mode}_{regime}_membrane.json", "w") as f:
                json.dump(_make_serializable(report), f, indent=2)
    else:
        print("[eval] No membrane recordings found - skipping membrane eval.")

    # Refresh the comparison tables over every run folder that exists in the study.
    study_dir = run_dir.parent
    existing_modes = sorted(
        d.name for d in study_dir.iterdir()
        if d.is_dir() and (d / "run_meta.json").exists()
    )
    write_summary_csv(study_dir, "vub_summary.csv", existing_modes, main_or_membrane="main")
    write_summary_csv(study_dir, "vub_membrane_summary.csv", existing_modes, main_or_membrane="membrane")

if __name__ == "__main__":
    main()
