from __future__ import annotations

import argparse
import json
import sys
import torch
import mlflow
import numpy as np

from pathlib import Path
_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import transformer_suite as ts
from utils.hyperparameters_transformer import Hyperparameters
from utils.utils_transformer import normalize_device_string, setup_mlflow, setup_torch
from train_vub import (
    REGIMES,
    _make_serializable,
    build_split_loader,
    build_uuid_loader,
    evaluate_regimes,
    make_hyperparameters,
    membrane_uuids,
    resolve_membrane_paths,
    resolve_paths,
    write_summary_csv,
)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune a V/U/B model on membrane recordings.")
    p.add_argument("--base-run", type=str, default=None,
                   help="Run dir from train_vub.py (contains model_final.pt).")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Direct checkpoint path (overrides --base-run).")
    p.add_argument("--study-dir", type=str, default="experiments/vub_ft_membrane")
    p.add_argument("--mode", type=str, default="both", choices=["voiced", "unvoiced", "both"])
    # data
    p.add_argument("--data-path", type=str, default=None, help="Main dataset root (for val/test eval).")
    p.add_argument("--voiced-path", type=str, default=None)
    p.add_argument("--unvoiced-path", type=str, default=None)
    p.add_argument("--textgrid-path", type=str, default=None)
    p.add_argument("--membrane-data-path", type=str, default=None)
    p.add_argument("--membrane-voiced-path", type=str, default=None)
    p.add_argument("--membrane-unvoiced-path", type=str, default=None)
    p.add_argument("--membrane-textgrid-path", type=str, default=None)
    p.add_argument("--train-uuids", type=str, default=None,
                   help="Explicit membrane UUIDs to train on.")
    p.add_argument("--eval-uuids", type=str, default=None,
                   help="Explicit membrane UUIDs to hold out for evaluation.")
    p.add_argument("--holdout-n", type=int, default=0,
                   help="Randomly hold out N membrane UUIDs for eval (seed from --seed).")
    # fine-tune training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the fine-tune.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=3401)
    p.add_argument("--both-voiced-weight", type=float, default=1.0)
    p.add_argument("--both-unvoiced-weight", type=float, default=1.0)
    p.add_argument("--unvoiced-curriculum-epochs", type=int, default=0)
    p.add_argument("--early-stop-source", type=str, default="both",
                   choices=["both", "voiced", "unvoiced"],
                   help="Which input view's validation loss drives early stopping.")
    p.add_argument("--save-checkpoints", action="store_true")
    # eval knobs
    p.add_argument("--vocode-samples", type=int, default=10)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--no-waveform-metrics", action="store_true")
    p.add_argument("--compute-efficiency", action="store_true")
    # env
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--mlflow-server-url", type=str, default=None)
    p.add_argument("--experiment-group", type=str, default="vub_finetune")
    return p

def main() -> None:
    args = build_parser().parse_args()

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        base_name = checkpoint.parent.name
    elif args.base_run:
        base_dir = Path(args.base_run)
        checkpoint = base_dir / "model_final.pt"
        base_name = base_dir.name
    else:
        sys.exit("Provide --base-run (a train_vub.py run dir) or --checkpoint.")
    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}")

    # Device / torch setup.
    hp0 = Hyperparameters()
    if args.cuda_device is not None:
        hp0.device = f"cuda:{args.cuda_device}"
    hp0.device = normalize_device_string(hp0.device)
    hp0.device = setup_torch(hp0.device)

    hp = make_hyperparameters(args, hp0)
    hp.learning_rate = args.lr
    print(f"[ft] base={checkpoint} mode={args.mode} lr={hp.learning_rate} "
          f"epochs={hp.epochs} device={hp.device}")

    # Main dataset paths (for val + test).
    voiced_path, unvoiced_path, textgrid_path = resolve_paths(args)

    # Membrane paths (training + optional held-out eval).
    base, m_voiced, m_unvoiced, m_textgrid = resolve_membrane_paths(args)
    def _has_files(d):
        return d.is_dir() and any(d.iterdir())
    if not (_has_files(m_voiced) and _has_files(m_unvoiced) and _has_files(m_textgrid)):
        sys.exit(f"Membrane recordings incomplete under {base}. "
                 "Need voiced_samples/, unvoiced_samples/, textgrids/ with files. "
                 "Set --membrane-data-path to the right location.")

    # Membrane UUID splits.
    all_uuids = membrane_uuids(args)
    train_uuids = [u for u in all_uuids]
    eval_uuids: list[str] = []
    if args.train_uuids:
        train_uuids = [u.strip() for u in args.train_uuids.split(",") if u.strip()]
        eval_uuids = [u for u in all_uuids if u not in train_uuids]
    if args.eval_uuids:
        eval_uuids = [u.strip() for u in args.eval_uuids.split(",") if u.strip()]
        train_uuids = [u for u in train_uuids if u not in eval_uuids]
    if args.holdout_n and args.holdout_n > 0 and not eval_uuids:
        rng = np.random.RandomState(args.seed)
        rng.shuffle(train_uuids)
        eval_uuids = train_uuids[-args.holdout_n:]
        train_uuids = train_uuids[:-args.holdout_n]
    if not train_uuids:
        sys.exit("No membrane UUIDs left for training - check --train-uuids/--eval-uuids/--holdout-n.")

    print(f"[ft] membrane: {len(train_uuids)} train, {len(eval_uuids)} held-out eval")

    # Loaders.
    train_loader = build_uuid_loader(hp, m_voiced, m_unvoiced, m_textgrid, train_uuids, shuffle=True)
    val_loader = build_split_loader(hp, voiced_path, unvoiced_path, textgrid_path, split="val")
    test_loader = build_split_loader(hp, voiced_path, unvoiced_path, textgrid_path, split="test")
    print(f"[ft] membrane train={len(train_loader.dataset)} | main val={len(val_loader.dataset)} "
          f"| main test={len(test_loader.dataset)}")

    run_dir = Path(args.study_dir) / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)

    experiment = setup_mlflow(args.mlflow_server_url, args.experiment_group) if args.mlflow_server_url else mlflow.set_experiment(args.experiment_group)
    mlflow.start_run(run_name=f"ft_{base_name}_{args.mode}", experiment_id=experiment.experiment_id)
    mlflow.set_tag("vub_mode", args.mode)
    mlflow.set_tag("finetune_base", str(checkpoint))
    mlflow.log_param("lr", hp.learning_rate)
    mlflow.log_param("epochs", hp.epochs)
    mlflow.log_param("train_uuids", len(train_uuids))
    mlflow.log_param("eval_uuids", len(eval_uuids))

    model, optimizer, scheduler = ts.init_model_and_optimizer(hp, checkpoint_path=str(checkpoint))
    mssc_loss_fn = ts.init_mssc_loss_fn(hp)

    history = ts.run_training_loop(
        model, optimizer, scheduler, train_loader, val_loader, test_loader,
        hp, mssc_loss_fn,
        train_on=args.mode, plot_every=0,
        history_csv_path=run_dir / "training_history.csv",
    )

    # Main test set under both input regimes.
    reports_main = evaluate_regimes(model, hp, test_loader, REGIMES, args, history)
    for regime, report in reports_main.items():
        with open(run_dir / f"metrics_{args.mode}_{regime}.json", "w") as f:
            json.dump(_make_serializable(report), f, indent=2)

    # Membrane held-out eval (only if we held some UUIDs out of training).
    if eval_uuids:
        membrane_eval_loader = build_uuid_loader(hp, m_voiced, m_unvoiced, m_textgrid, eval_uuids, shuffle=False)
        reports_membrane = evaluate_regimes(model, hp, membrane_eval_loader, REGIMES, args, history)
        for regime, report in reports_membrane.items():
            with open(run_dir / f"metrics_{args.mode}_{regime}_membrane.json", "w") as f:
                json.dump(_make_serializable(report), f, indent=2)

    torch.save({"model_state_dict": model.state_dict()}, run_dir / "model_final.pt")

    meta = {
        "mode": args.mode,
        "finetune_from": str(checkpoint),
        "best_epoch": int(history.get("best_epoch", -1)),
        "best_val_loss": float(history.get("best_val_loss", float("nan"))),
        "epochs_run": int(len(history.get("epoch", []))),
        "membrane_train_uuids": train_uuids,
        "membrane_eval_uuids": eval_uuids,
        "checkpoint": str(run_dir / "model_final.pt"),
        "seed": int(hp.seed),
        "learning_rate": float(hp.learning_rate),
        "both_voiced_weight": float(hp.both_voiced_weight),
        "both_unvoiced_weight": float(hp.both_unvoiced_weight),
        "unvoiced_curriculum_epochs": int(hp.unvoiced_curriculum_epochs),
        "early_stop_source": str(hp.early_stop_source),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    mlflow.end_run()
    print(f"[ft] Finished -> {run_dir}")

    # Refresh comparison tables for whatever runs live in this study dir.
    existing = sorted(d.name for d in Path(args.study_dir).iterdir()
                      if d.is_dir() and (d / "run_meta.json").exists())
    write_summary_csv(Path(args.study_dir), "vub_summary.csv", existing, main_or_membrane="main")
    write_summary_csv(Path(args.study_dir), "vub_membrane_summary.csv", existing, main_or_membrane="membrane")

if __name__ == "__main__":
    main()