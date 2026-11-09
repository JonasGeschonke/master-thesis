from __future__ import annotations

import argparse
import csv
import json
import sys
import torch
import mlflow
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import transformer_suite as ts
from ablation import MaskedDataset, build_run_specs
from dataset.transformer_dataset import SSIDataset, pad_collate_fn
from metrics.metrics_suite import MetricsSuite
from utils.hyperparameters_transformer import Hyperparameters
from utils.utils_transformer import (
    normalize_device_string,
    save_checkpoint,
    setup_manual_seed,
    setup_mlflow,
    setup_torch,
)

BACKEND_URL = ts.BACKEND_URL
MLFLOW_SERVER_URL = ts.MLFLOW_SERVER_URL
DATA_PATH = ts.DATA_PATH
UNVOICED_PATH = ts.UNVOICED_PATH
VOICED_PATH = ts.VOICED_PATH
TEXTGRID_PATH = ts.TEXTGRID_PATH

HEADLINE_METRICS = {
    "mcd_mean": ("spectral", "mcd_mean"),
    "lsd_mean": ("spectral", "lsd_mean"),
    "mel_rmse_mean": ("spectral", "mel_rmse_mean"),
    "pearson_mean": ("spectral", "pearson_mean"),
    "stoi_mean": ("perceptual", "stoi_mean"),
    "pesq_mean": ("perceptual", "pesq_mean"),
    "dnsmos_mean": ("perceptual", "dnsmos_mean"),
    "waveform_mse_mean": ("perceptual", "waveform_mse_mean"),
    "per_mean": ("intelligibility", "per_mean"),
    "wer_mean": ("intelligibility", "wer_mean"),
    "cer_mean": ("intelligibility", "cer_mean"),
    "dtw_cost_mean": ("ssi_specific", "dtw_cost_mean"),
    "f0_corr_mean": ("ssi_specific", "f0_corr_mean"),
    "voiced_unvoiced_acc_mean": ("ssi_specific", "voiced_unvoiced_acc_mean"),
    "duration_corr_mean": ("ssi_specific", "duration_corr_mean"),
}

HIGHER_IS_BETTER = {
    "pearson_mean", "stoi_mean", "pesq_mean", "dnsmos_mean",
    "f0_corr_mean", "voiced_unvoiced_acc_mean", "duration_corr_mean",
}

IMPORTANCE_PLOTS = {
    "mel_rmse_mean": "Mel RMSE",
    "per_mean": "PER",
    "stoi_mean": "STOI",
}

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full feature ablation study (retrain per feature).")
    p.add_argument("--data-path", type=str, default=str(DATA_PATH))
    p.add_argument("--voiced-path", type=str, default=None)
    p.add_argument("--unvoiced-path", type=str, default=None)
    p.add_argument("--textgrid-path", type=str, default=None)
    p.add_argument("--study-dir", type=str, default="ablation_study")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-on", type=str, default="voiced", choices=["voiced", "unvoiced", "both"])
    p.add_argument("--seed", type=int, default=3401)
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--vocode-samples", type=int, default=10)
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated substrings matching feature names to run (e.g. baseline,subband).")
    p.add_argument("--list-features", action="store_true", help="Print the run list and exit.")
    p.add_argument("--strict-deterministic", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--skip-aggregation", action="store_true")
    p.add_argument("--mlflow-server-url", type=str, default=MLFLOW_SERVER_URL)
    p.add_argument("--experiment-group", type=str, default="ablation_study")
    return p

def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

def _make_serializable(obj):
    """Recursively convert a report into JSON-serializable primitives."""
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

def build_masked_loaders(spec, hyperparameters, voiced_path, unvoiced_path, textgrid_path, debug):
    pin_memory = isinstance(hyperparameters.device, str) and hyperparameters.device.startswith("cuda")
    loaders = {}
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        dataset = SSIDataset(hyperparameters, split=split, debug=debug)
        dataset._load_data(voiced_path, unvoiced_path, textgrid_path)
        dataset = MaskedDataset(dataset, spec)
        loaders[split] = DataLoader(
            dataset,
            pin_memory=pin_memory,
            batch_size=hyperparameters.batch_size,
            shuffle=shuffle,
            collate_fn=pad_collate_fn,
            num_workers=hyperparameters.num_workers,
        )
    return loaders["train"], loaders["val"], loaders["test"]

def _save_confusion_matrix(report: dict, run_dir: Path) -> None:
    cm = report.get("intelligibility", {}).get("confusion_matrix")
    if cm is None:
        return
    labels = report["intelligibility"].get("confusion_labels", [])
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Phoneme Confusion Matrix ({len(labels)} classes)")
    if labels:
        step = max(1, len(labels) // 40)
        pos = list(range(0, len(labels), step))
        ax.set_xticks(pos)
        ax.set_yticks(pos)
        ax.set_xticklabels([labels[i] for i in pos], fontsize=7, rotation=90)
        ax.set_yticklabels([labels[i] for i in pos], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def run_one(spec, hyperparameters, experiment, args, study_dir, voiced_path, unvoiced_path, textgrid_path):
    run_dir = study_dir / _sanitize(spec.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    hyperparameters.save_dir = str(run_dir)

    setup_manual_seed(hyperparameters.seed)

    train_loader, val_loader, test_loader = build_masked_loaders(
        spec, hyperparameters, voiced_path, unvoiced_path, textgrid_path, args.debug,
    )

    model, optimizer, scheduler = ts.init_model_and_optimizer(hyperparameters, checkpoint_path=None)
    mssc_loss_fn = ts.init_mssc_loss_fn(hyperparameters)

    mlflow.start_run(run_name=f"ablation_{spec.name}", experiment_id=experiment.experiment_id)
    try:
        mlflow.set_tag("ablation_feature", spec.name)
        mlflow.set_tag("stream", spec.stream)
        mlflow.set_tag("run_type", spec.run_type)
        mlflow.set_tag("train_on", args.train_on)
        mlflow.log_param("epochs", hyperparameters.epochs)
        mlflow.log_param("seed", hyperparameters.seed)
        mlflow.log_param("batch_size", hyperparameters.batch_size)

        history = ts.run_training_loop(
            model, optimizer, scheduler, train_loader, val_loader, test_loader,
            hyperparameters, mssc_loss_fn,
            train_on=args.train_on, plot_every=0,
            history_csv_path=run_dir / "training_history.csv",
        )

        # Cheap test-set scalars (prefix "test").
        ts.run_evaluation(model, test_loader, hyperparameters, mssc_loss_fn)

        # Full 6-group metrics suite (scalars → MLflow, no artifacts).
        suite = MetricsSuite(
            model=model,
            hyperparameters=hyperparameters,
            phoneme_inventory=ts.PHONEME_INVENTORY,
            checkpoint_path=None,
        )
        report = suite.evaluate(
            test_loader,
            max_batches=None,
            train_on=args.train_on,
            vocode_samples=args.vocode_samples,
            training_history=history,
        )
        suite.log_to_mlflow(report, step=hyperparameters.epochs)
        print(suite.summary(report))

        # Local artifacts: metrics.json, confusion matrix, example plots.
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(_make_serializable(report), f, indent=2)
        _save_confusion_matrix(report, run_dir)

        last_epoch = int(history.get("best_epoch", hyperparameters.epochs - 1))
        ts.create_plots(
            model, val_loader, hyperparameters,
            epoch=last_epoch, train_on=args.train_on, plot_dir_name="plots",
        )

        if args.save_checkpoints:
            save_checkpoint(
                model, optimizer, hyperparameters.epochs,
                history.get("best_val_loss", 0.0),
                run_dir / "checkpoints", scheduler=scheduler,
            )

        print(f"[run] Finished {spec.name}: {run_dir}")
        return history, report
    finally:
        mlflow.end_run()

def aggregate_results(study_dir: Path) -> None:
    spec_by_name = {s.name: s for s in build_run_specs()}
    rows = []
    baseline = None

    for metrics_file in sorted(study_dir.glob("*/metrics.json")):
        run_id = metrics_file.parent.name
        try:
            with open(metrics_file) as f:
                report = json.load(f)
        except Exception as exc:
            print(f"[aggregate] Skipping {run_id}: {exc}")
            continue

        spec = spec_by_name.get(run_id)
        row: dict = {
            "feature": run_id,
            "stream": spec.stream if spec else "",
            "run_type": spec.run_type if spec else "",
        }
        td = report.get("training_dynamics", {})
        row["epochs_run"] = td.get("epochs_run")
        row["best_epoch"] = td.get("best_epoch")
        row["best_val_loss"] = td.get("best_val_loss")
        row["convergence_epoch"] = td.get("convergence_epoch")
        row["generalization_gap"] = td.get("generalization_gap")

        for key, (group, sub) in HEADLINE_METRICS.items():
            val = report.get(group, {}).get(sub)
            row[key] = val if isinstance(val, (int, float)) else float("nan")

        rows.append(row)
        if run_id == "baseline":
            baseline = row

    if not rows:
        print("[aggregate] No metrics.json files found.")
        return

    columns = [
        "feature", "stream", "run_type", "epochs_run", "best_epoch", "best_val_loss",
        "convergence_epoch", "generalization_gap",
    ] + list(HEADLINE_METRICS.keys())

    csv_path = study_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[aggregate] Wrote {csv_path} ({len(rows)} rows)")

    if baseline is None:
        print("[aggregate] No baseline run found; skipping importance plots.")
        return

    for key, label in IMPORTANCE_PLOTS.items():
        _importance_bar_chart(key, label, rows, baseline, study_dir)

def _importance_bar_chart(metric_key: str, label: str, rows, baseline, study_dir: Path) -> None:
    entries = []
    for r in rows:
        if r["feature"] == "baseline":
            continue
        b = baseline.get(metric_key)
        v = r.get(metric_key)
        if b is None or v is None or not np.isfinite(b) or not np.isfinite(v):
            continue
        delta = (v - b) if metric_key not in HIGHER_IS_BETTER else (b - v)
        entries.append((r["feature"], float(delta)))

    if not entries:
        print(f"[aggregate] No finite values for {metric_key}; skipping plot.")
        return

    entries.sort(key=lambda x: x[1], reverse=True)
    names = [e[0] for e in entries]
    vals = [e[1] for e in entries]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(entries))))
    y = np.arange(len(entries))
    ax.barh(y, vals, color="#4472C4")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Importance (Δ {label})")
    ax.set_title(f"Feature Importance: {label} (leave-one-out)")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    fig.savefig(study_dir / f"feature_importance_{metric_key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate] Wrote {study_dir / f'feature_importance_{metric_key}.png'}")

def main() -> None:
    args = build_parser().parse_args()
    specs = build_run_specs()

    if args.list_features:
        for s in specs:
            print(f"{s.run_type:9s} {s.name}")
        return

    if args.only:
        tokens = [t.strip() for t in args.only.split(",") if t.strip()]
        specs = [s for s in specs if any(t in s.name for t in tokens)]
        if not specs:
            print("No features matched --only filter.")
            return

    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    voiced_path = Path(args.voiced_path) if args.voiced_path else data_path / "voiced_samples"
    unvoiced_path = Path(args.unvoiced_path) if args.unvoiced_path else data_path / "unvoiced_samples"
    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else data_path / "textgrids"

    # Device / torch setup once, shared by every run.
    hp0 = Hyperparameters()
    if args.cuda_device is not None:
        hp0.device = f"cuda:{args.cuda_device}"
    hp0.device = normalize_device_string(hp0.device)
    hp0.device = setup_torch(hp0.device)
    if args.strict_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch without the warn_only kwarg — enable strictly.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:
                print(f"[setup] Could not enable deterministic algorithms: {exc}")

    ts.TRAIN_ON = args.train_on

    experiment = setup_mlflow(args.mlflow_server_url, args.experiment_group)
    print(f"Running {len(specs)} runs into {study_dir.resolve()}")

    for spec in specs:
        hyperparameters = Hyperparameters()
        hyperparameters.epochs = args.epochs
        hyperparameters.batch_size = args.batch_size
        hyperparameters.seed = args.seed
        hyperparameters.device = hp0.device
        print(f"\n=== Run: {spec.name} ({spec.run_type}) ===")
        run_one(spec, hyperparameters, experiment, args, study_dir, voiced_path, unvoiced_path, textgrid_path)

    if not args.skip_aggregation:
        aggregate_results(study_dir)

if __name__ == "__main__":
    main()
