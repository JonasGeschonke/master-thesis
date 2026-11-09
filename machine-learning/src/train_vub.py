from __future__ import annotations

import argparse
import csv
import json
import sys
import numpy as np
import torch
import mlflow

from pathlib import Path
from torch.utils.data import DataLoader
_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import transformer_suite as ts
from metrics.metrics_suite import MetricsSuite
from utils.hyperparameters_transformer import Hyperparameters
from utils.utils_transformer import (
    normalize_device_string,
    save_checkpoint,
    setup_manual_seed,
    setup_mlflow,
    setup_torch,
)
from dataset.transformer_dataset import SSIDataset, pad_collate_fn

MODES = ["voiced", "unvoiced", "both"]
REGIMES = ["voiced", "unvoiced"]

# Headline metrics for the compact comparison table (key -> (group, subkey)).
HEADLINE_METRICS = {
    "mel_rmse_mean": ("spectral", "mel_rmse_mean"),
    "lsd_mean": ("spectral", "lsd_mean"),
    "mcd_mean": ("spectral", "mcd_mean"),
    "pearson_mean": ("spectral", "pearson_mean"),
    "stoi_mean": ("perceptual", "stoi_mean"),
    "pesq_mean": ("perceptual", "pesq_mean"),
    "waveform_mse_mean": ("perceptual", "waveform_mse_mean"),
    "per_mean": ("intelligibility", "per_mean"),
    "wer_mean": ("intelligibility", "wer_mean"),
    "cer_mean": ("intelligibility", "cer_mean"),
    "dtw_cost_mean": ("ssi_specific", "dtw_cost_mean"),
    "f0_corr_mean": ("ssi_specific", "f0_corr_mean"),
    "voiced_unvoiced_acc_mean": ("ssi_specific", "voiced_unvoiced_acc_mean"),
    "duration_corr_mean": ("ssi_specific", "duration_corr_mean"),
}

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V/U/B training study for the SSI transformer.")
    # data
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--voiced-path", type=str, default=None)
    p.add_argument("--unvoiced-path", type=str, default=None)
    p.add_argument("--textgrid-path", type=str, default=None)
    # membrane held-out eval (optional)
    p.add_argument("--membrane-data-path", type=str, default=None)
    p.add_argument("--membrane-voiced-path", type=str, default=None)
    p.add_argument("--membrane-unvoiced-path", type=str, default=None)
    p.add_argument("--membrane-textgrid-path", type=str, default=None)
    p.add_argument("--membrane-uuids", type=str, default=None,
                   help="Comma-separated UUIDs; default = MEMBRANE_RECORDING_UUIDS")
    # study / runs
    p.add_argument("--study-dir", type=str, default="experiments/vub")
    p.add_argument("--modes", type=str, default="voiced,unvoiced,both",
                   help="Which training modes to run (comma separated).")
    # training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=3401)
    p.add_argument("--both-voiced-weight", type=float, default=1.0,
                   help="Loss weight for the voiced view when --modes includes 'both'.")
    p.add_argument("--both-unvoiced-weight", type=float, default=1.0,
                   help="Loss weight for the unvoiced view when --modes includes 'both'.")
    p.add_argument("--unvoiced-curriculum-epochs", type=int, default=0,
                   help=">0: ramp the unvoiced view weight from 0 to --both-unvoiced-weight over the first N epochs (voiced-first curriculum).")
    p.add_argument("--early-stop-source", type=str, default="both",
                   choices=["both", "voiced", "unvoiced"],
                   help="Which input view's validation loss drives early stopping (default: averaged over both).")
    p.add_argument("--save-checkpoints", action="store_true")
    # eval knobs
    p.add_argument("--vocode-samples", type=int, default=10)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--no-waveform-metrics", action="store_true",
                   help="Skip vocoding-dependent metrics (STOI/PESQ/WER/CER...).")
    p.add_argument("--compute-efficiency", action="store_true")
    # environment
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--mlflow-server-url", type=str, default=None,
                   help="MLflow tracking URI; default = local ./mlruns file store.")
    p.add_argument("--experiment-group", type=str, default="vub_experiments")
    return p

def _make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return str(obj)

def resolve_paths(args) -> tuple[Path, Path, Path]:
    """Return (voiced_path, unvoiced_path, textgrid_path) for the MAIN dataset."""
    if args.data_path is None:
        data_path = _SCRIPT_ROOT.parent / "transformer_data"
    else:
        data_path = Path(args.data_path)
    voiced = Path(args.voiced_path) if args.voiced_path else data_path / "voiced_samples"
    unvoiced = Path(args.unvoiced_path) if args.unvoiced_path else data_path / "unvoiced_samples"
    textgrid = Path(args.textgrid_path) if args.textgrid_path else data_path / "textgrids"
    return voiced, unvoiced, textgrid

def resolve_membrane_paths(args):
    """Return membrane (voiced, unvoiced, textgrid) dirs or None if unused."""
    if args.membrane_data_path is None:
        base = (Path(args.data_path) if args.data_path else _SCRIPT_ROOT.parent / "transformer_data") / "membrane"
    else:
        base = Path(args.membrane_data_path)
    voiced = Path(args.membrane_voiced_path) if args.membrane_voiced_path else base / "voiced_samples"
    unvoiced = Path(args.membrane_unvoiced_path) if args.membrane_unvoiced_path else base / "unvoiced_samples"
    textgrid = Path(args.membrane_textgrid_path) if args.membrane_textgrid_path else base / "textgrids"
    return base, voiced, unvoiced, textgrid

def build_split_loader(hp, voiced_path, unvoiced_path, textgrid_path, split: str = "test"):
    """Build ONLY one deterministic split (train/val/test) - no other splits.

    The split only depends on the file list + hp.seed (+ train/val ratios), so
    this reproduces the exact split a training run used — cheap for eval/finetune.
    """
    setup_manual_seed(hp.seed)
    dataset = SSIDataset(hp, split=split, debug=False)
    dataset._load_data(voiced_path, unvoiced_path, textgrid_path)
    return DataLoader(
        dataset,
        pin_memory=isinstance(hp.device, str) and hp.device.startswith("cuda"),
        batch_size=hp.batch_size,
        shuffle=split == "train",
        collate_fn=pad_collate_fn,
        num_workers=hp.num_workers,
    )

def build_test_loader(hp, voiced_path, unvoiced_path, textgrid_path):
    """Deterministic test-split loader (matches the split used during training)."""
    return build_split_loader(hp, voiced_path, unvoiced_path, textgrid_path, split="test")


def build_uuid_loader(hp, voiced_path, unvoiced_path, textgrid_path, uuids, shuffle=False):
    """Load ONLY the given UUID recordings (voiced+unvoiced+textgrid triplets)."""
    dataset = SSIDataset(hp, split="train" if shuffle else "test", debug=False)
    dataset.load_files(
        voiced_path=voiced_path,
        unvoiced_path=unvoiced_path,
        textgrid_path=textgrid_path,
        file_names=[f"{u}.flac" for u in uuids],
    )
    return DataLoader(
        dataset,
        pin_memory=isinstance(hp.device, str) and hp.device.startswith("cuda"),
        batch_size=hp.batch_size,
        shuffle=shuffle,
        collate_fn=pad_collate_fn,
        num_workers=hp.num_workers,
    )

def evaluate_regimes(model, hp, loader, regimes, args, history):
    """Run the full 6-group MetricsSuite on `loader` under each input regime."""
    suite = MetricsSuite(
        model=model,
        hyperparameters=hp,
        phoneme_inventory=ts.PHONEME_INVENTORY,
        checkpoint_path=None,
    )
    reports = {}
    for regime in regimes:
        print(f"[eval] regime='{regime}'  loader={getattr(loader, 'dataset', loader)} samples={len(loader.dataset) if hasattr(loader, 'dataset') else '?'}")
        report = suite.evaluate(
            loader,
            max_batches=args.max_batches,
            compute_waveform_metrics=not args.no_waveform_metrics,
            compute_efficiency=args.compute_efficiency,
            vocode_samples=args.vocode_samples,
            training_history=history,
            train_on=regime,
        )
        reports[regime] = report
    return reports

def make_hyperparameters(args, hp0):
    """Build a fresh Hyperparameters with the run-level overrides applied."""
    hyperparameters = Hyperparameters()
    hyperparameters.epochs = args.epochs
    hyperparameters.batch_size = args.batch_size
    hyperparameters.seed = args.seed
    hyperparameters.device = hp0.device
    for _attr in ("both_voiced_weight", "both_unvoiced_weight",
                  "unvoiced_curriculum_epochs", "early_stop_source"):
        _v = getattr(args, _attr, None)
        if _v is not None:
            setattr(hyperparameters, _attr, _v)
    return hyperparameters

def membrane_uuids(args) -> list[str]:
    if args.membrane_uuids:
        return [u.strip() for u in args.membrane_uuids.split(",") if u.strip()]
    return list(ts.MEMBRANE_RECORDING_UUIDS)

def run_one_mode(mode, args, hp0, experiment, study_dir):
    run_dir = study_dir / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = make_hyperparameters(args, hp0)
    setup_manual_seed(hyperparameters.seed)
    voiced_path, unvoiced_path, textgrid_path = resolve_paths(args)
    train_loader, val_loader, test_loader = ts.load_datasets(
        voiced_path, unvoiced_path, textgrid_path, hyperparameters, debug=False,
    )

    print(f"\n=== [{mode}] training ({len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test) ===")

    mlflow.start_run(run_name=f"vub_{mode}", experiment_id=experiment.experiment_id)
    mlflow.set_tag("vub_mode", mode)
    mlflow.set_tag("input_branch", mode)
    mlflow.log_param("epochs", hyperparameters.epochs)
    mlflow.log_param("batch_size", hyperparameters.batch_size)
    mlflow.log_param("seed", hyperparameters.seed)

    model, optimizer, scheduler = ts.init_model_and_optimizer(hyperparameters, checkpoint_path=None)
    mssc_loss_fn = ts.init_mssc_loss_fn(hyperparameters)

    history = ts.run_training_loop(
        model, optimizer, scheduler, train_loader, val_loader, test_loader,
        hyperparameters, mssc_loss_fn,
        train_on=mode, plot_every=0,
        history_csv_path=run_dir / "training_history.csv",
    )

    # ----- main test set, evaluated under both input regimes -----
    reports_main = evaluate_regimes(model, hyperparameters, test_loader, REGIMES, args, history)
    for regime, report in reports_main.items():
        with open(run_dir / f"metrics_{mode}_{regime}.json", "w") as f:
            json.dump(_make_serializable(report), f, indent=2)
        try:
            MetricsSuite(model, hyperparameters,
                         phoneme_inventory=ts.PHONEME_INVENTORY).log_to_mlflow(report, step=hyperparameters.epochs)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! mlflow log failed for regime {regime}: {exc}")

    # ----- membrane held-out set (optional) -----
    reports_membrane = None
    base, m_voiced, m_unvoiced, m_textgrid = resolve_membrane_paths(args)
    if base.exists() and any(p.exists() for p in (m_voiced, m_unvoiced, m_textgrid)):
        membrane_loader = ts.load_membrane_test_loader(
            hyperparameters, m_voiced, m_unvoiced, m_textgrid, membrane_uuids(args), debug=False,
        )
        reports_membrane = evaluate_regimes(model, hyperparameters, membrane_loader, REGIMES, args, history)
        for regime, report in reports_membrane.items():
            with open(run_dir / f"metrics_{mode}_{regime}_membrane.json", "w") as f:
                json.dump(_make_serializable(report), f, indent=2)

    # ----- persist checkpoint + metadata -----
    torch_model_path = run_dir / "model_final.pt"
    torch.save({"model_state_dict": model.state_dict()}, torch_model_path)
    if args.save_checkpoints:
        save_checkpoint(model, optimizer, hyperparameters.epochs,
                        history.get("best_val_loss", 0.0),
                        run_dir / "checkpoints", scheduler=scheduler)

    meta = {
        "mode": mode,
        "best_epoch": int(history.get("best_epoch", -1)),
        "best_val_loss": float(history.get("best_val_loss", float("nan"))),
        "epochs_run": int(len(history.get("epoch", []))),
        "checkpoint": str(torch_model_path),
        "study_dir": str(run_dir),
        "membrane_evaluated": reports_membrane is not None,
        "seed": int(hyperparameters.seed),
        "batch_size": int(hyperparameters.batch_size),
        "epochs_requested": int(hyperparameters.epochs),
        "learning_rate": float(hyperparameters.learning_rate),
        "both_voiced_weight": float(hyperparameters.both_voiced_weight),
        "both_unvoiced_weight": float(hyperparameters.both_unvoiced_weight),
        "unvoiced_curriculum_epochs": int(hyperparameters.unvoiced_curriculum_epochs),
        "early_stop_source": str(hyperparameters.early_stop_source),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if mlflow.active_run() is not None:
        mlflow.end_run()
    print(f"[run] Finished mode={mode}: {run_dir}")
    return mode, reports_main, reports_membrane

# ---------------------------------------------------------------------------
# Compact comparison tables
# ---------------------------------------------------------------------------

def _metric(report, key):
    group, sub = HEADLINE_METRICS[key]
    try:
        return float(report.get(group, {}).get(sub, float("nan")))
    except (TypeError, ValueError):
        return float("nan")

def write_summary_csv(study_dir: Path, fname: str, modes: list[str], main_or_membrane: str = "main") -> None:
    """Write one row per (mode x regime): mode, regime, then headline metrics."""
    suffix = "" if main_or_membrane == "main" else "_membrane"
    columns = ["mode", "regime", "best_epoch", "best_val_loss"] + list(HEADLINE_METRICS.keys())
    rows = []
    for mode in modes:
        run_dir = study_dir / mode
        meta = {}
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        for regime in REGIMES:
            report_file = run_dir / f"metrics_{mode}_{regime}{suffix}.json"
            if not report_file.exists():
                continue
            with open(report_file) as f:
                report = json.load(f)
            row = {"mode": mode, "regime": regime,
                   "best_epoch": meta.get("best_epoch"),
                   "best_val_loss": meta.get("best_val_loss")}
            for key in HEADLINE_METRICS:
                row[key] = _metric(report, key)
            rows.append(row)

    out = study_dir / fname
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[summary] Wrote {out} ({len(rows)} rows)")

    if rows:
        print("\nCompact view (MCD | mel-RMSE | PER | STOI):")
        print(f"{'mode':9s} {'regime':9s} {'MCD':>10s} {'melRMSE':>9s} {'PER':>7s} {'STOI':>7s}")
        for r in rows:
            def fmt(v):
                return "  nan" if v != v else f"{v:9.4f}"  # NaN check
            print(f"{r['mode']:9s} {r['regime']:9s} {fmt(r.get('mcd_mean', float('nan')))} "
                  f"{fmt(r.get('mel_rmse_mean', float('nan')))} "
                  f"{fmt(r.get('per_mean', float('nan')))} "
                  f"{fmt(r.get('stoi_mean', float('nan')))}")

def main() -> None:
    args = build_parser().parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in MODES]
    if invalid:
        sys.exit(f"Invalid --modes: {invalid} (choose from {MODES})")

    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    # Shared device / torch setup once.
    hp0 = Hyperparameters()
    if args.cuda_device is not None:
        hp0.device = f"cuda:{args.cuda_device}"
    hp0.device = normalize_device_string(hp0.device)
    hp0.device = setup_torch(hp0.device)
    print(f"[setup] device = {hp0.device}")

    experiment = None
    if args.mlflow_server_url:
        experiment = setup_mlflow(args.mlflow_server_url, args.experiment_group)
    else:
        experiment = mlflow.set_experiment(args.experiment_group)

    completed = []
    for mode in modes:
        mode, _, _ = run_one_mode(mode, args, hp0, experiment, study_dir)
        completed.append(mode)

    # main test table + membrane table
    write_summary_csv(study_dir, "vub_summary.csv", completed, main_or_membrane="main")
    write_summary_csv(study_dir, "vub_membrane_summary.csv", completed, main_or_membrane="membrane")
    print(f"\nAll done -> {study_dir.resolve()}")

if __name__ == "__main__":
    main()