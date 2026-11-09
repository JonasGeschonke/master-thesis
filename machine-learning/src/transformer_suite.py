import sys
import os
import torch
import numpy as np
import mlflow

from datetime import datetime
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mlflow.pytorch import log_model
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW

from criterion.multi_scale_mel_loss import MultiScaleMelLoss
from dataset.transformer_dataset import SSIDataset, pad_collate_fn
from utils.hyperparameters_transformer import Hyperparameters
from utils.utils_transformer import PHONEME_INVENTORY, download_data, download_data_for_uuids, setup_manual_seed, setup_torch, setup_mlflow, save_checkpoint, normalize_device_string
from models.transformer import TransformerModel
from metrics.metrics_suite import MetricsSuite, evaluate_model

script_root = Path(__file__).resolve().parent
sys.path.insert(0, str(script_root))

BACKEND_URL = "https://ssi-gatherer.pb.ax"
MLFLOW_SERVER_URL = "https://mlflow.pb.ax"
DATA_PATH = Path("transformer_data")
UNVOICED_PATH = DATA_PATH / "unvoiced_samples"
VOICED_PATH = DATA_PATH / "voiced_samples"
TEXTGRID_PATH = DATA_PATH / "textgrids"
TRAIN_ON = "voiced"  # Options: "voiced", "unvoiced", "both"

DOWNLOAD_DATA = False
DEBUG = False
START_FROM_CHECKPOINT = False

MEMBRANE_RECORDING_UUIDS = [
    'd77add93-b24d-49a5-840e-b38d9f0057b0',
    '111d7799-b14e-498e-be1d-86df1eff2e80',
    '67abc685-5d91-42cc-8f83-f73a245d6da4',
    '0b3745b8-70fb-4d72-b2ee-ff81f15f2159',
    'ddeeee44-bb0a-4f2a-8a5f-19596d2008ee',
    'e0177f61-e813-459d-9b1d-40608a81c626',
    'd4e3504d-c74f-478b-8292-81f8ce01762f',
    '49c6a592-4284-484d-8970-ea1c7090a734',
    '42af7529-9b17-4f0a-81be-fe7ca18c0bff',
    'e661eb93-862a-4e6f-bb9e-f0a741a0da81'
]

hyperparameters = Hyperparameters()

def build_cli_parser():
    parser = ArgumentParser(description="Train and evaluate the Transformer model for SSI.")
    parser.add_argument("--download-data", action="store_true", help="Download dataset from remote server.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited data.")
    parser.add_argument("--start-from-checkpoint", action="store_true", help="Start training from a saved checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to the checkpoint file to load.")
    parser.add_argument("--save-dir", type=str, default='./checkpoints', help="Directory to save plots and images created during training.")
    parser.add_argument("--mlflow-server-url", type=str, default=MLFLOW_SERVER_URL, help="MLflow server URL.")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH), help="Path to the dataset directory.")
    parser.add_argument("--unvoiced-path", type=str, default=str(UNVOICED_PATH), help="Path to the unvoiced samples directory.")
    parser.add_argument("--voiced-path", type=str, default=str(VOICED_PATH), help="Path to the voiced samples directory.")
    parser.add_argument("--textgrid-path", type=str, default=str(TEXTGRID_PATH), help="Path to the textgrid directory.")
    parser.add_argument("--epochs", type=int, default=hyperparameters.epochs, help="Number of training epochs.")
    parser.add_argument("--backend-url", type=str, default=BACKEND_URL, help="Remote backend URL for downloading data.")
    parser.add_argument("--batch_size", type=int, default=16, help="Specify the batch size for this run.")
    parser.add_argument("--train-on", type=str, default="voiced", choices=["voiced", "unvoiced", "both"], help="Specify which data to train on: voiced, unvoiced, or both.")
    parser.add_argument("--run_name", type=str, default=f"experiment-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", help="Name of this experiment run.")
    parser.add_argument("--experiment-group", type=str, default=hyperparameters.experiment_name, help="Name of the experiment-group for MLflow tracking.")
    parser.add_argument("--full-metrics", action="store_true", help="Run the full 6-group metrics suite (specs/metrics.md) on the test set after training.")
    parser.add_argument("--metrics-only", action="store_true", help="Skip training and only run the metrics suite on the test set (requires --checkpoint-path).")
    parser.add_argument("--metrics-max-batches", type=int, default=None, help="Limit metrics evaluation to N batches (None = all).")
    parser.add_argument("--metrics-vocode-samples", type=int, default=10, help="Number of samples to vocode for waveform metrics.")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA visible device index to use, e.g. 0 if CUDA_VISIBLE_DEVICES=5.")
    # --- Membrane test-set evaluation ---
    parser.add_argument("--membrane-test", action="store_true", help="Evaluate --checkpoint-path on the membrane (new silent-speech technique) test set: runs the full 6-group metrics suite + plot generation. Standalone mode (no training).")
    parser.add_argument("--membrane-data-path", type=str, default="transformer_data/membrane", help="Base directory for the membrane test recordings (kept separate from the main dataset).")
    parser.add_argument("--membrane-voiced-path", type=str, default=None, help="Voiced (.flac) directory for membrane test recordings. Defaults to <--membrane-data-path>/voiced_samples.")
    parser.add_argument("--membrane-unvoiced-path", type=str, default=None, help="Unvoiced (.flac) directory for membrane test recordings. Defaults to <--membrane-data-path>/unvoiced_samples.")
    parser.add_argument("--membrane-textgrid-path", type=str, default=None, help="TextGrid directory for membrane test recordings. Defaults to <--membrane-data-path>/textgrids.")
    # --- Add membrane unvoiced recordings to the main training dataset ---
    parser.add_argument("--add-membrane-to-training", action="store_true", help="Download membrane recordings (MEMBRANE_RECORDING_UUIDS) into the main data directories so they are included in the train/val/test split. The unvoiced recordings are added alongside the existing voiced data to test whether features from the new recording technique are comparable.")
    return parser

def load_datasets(voiced_path, unvoiced_path, textgrid_path, hyperparameters, debug=False):
    train_dataset = SSIDataset(hyperparameters, split='train', debug=debug)
    val_dataset = SSIDataset(hyperparameters, split='val', debug=debug)
    test_dataset = SSIDataset(hyperparameters, split='test', debug=debug)

    train_dataset._load_data(voiced_path, unvoiced_path, textgrid_path)
    val_dataset._load_data(voiced_path, unvoiced_path, textgrid_path)
    test_dataset._load_data(voiced_path, unvoiced_path, textgrid_path)

    train_loader = DataLoader(train_dataset, pin_memory=isinstance(hyperparameters.device, str) and hyperparameters.device.startswith('cuda'), batch_size=hyperparameters.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=hyperparameters.num_workers)
    val_loader = DataLoader(val_dataset, pin_memory=isinstance(hyperparameters.device, str) and hyperparameters.device.startswith('cuda'), batch_size=hyperparameters.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=hyperparameters.num_workers)
    test_loader = DataLoader(test_dataset, pin_memory=isinstance(hyperparameters.device, str) and hyperparameters.device.startswith('cuda'), batch_size=hyperparameters.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=hyperparameters.num_workers)

    return train_loader, val_loader, test_loader

def load_membrane_test_loader(hyperparameters, voiced_path, unvoiced_path, textgrid_path, uuids, debug=False):
    # Build a test-only DataLoader from the membrane recording UUIDs.
    dataset = SSIDataset(hyperparameters, split='test', debug=debug)
    dataset.load_files(
        voiced_path=voiced_path,
        unvoiced_path=unvoiced_path,
        textgrid_path=textgrid_path,
        file_names=[f"{uuid}.flac" for uuid in uuids],
    )
    print(f"[membrane] Membrane test set: {len(dataset)} recordings loaded (from {len(uuids)} requested UUIDs).")

    return DataLoader(
        dataset,
        pin_memory=isinstance(hyperparameters.device, str) and hyperparameters.device.startswith('cuda'),
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=hyperparameters.num_workers,
    )

def init_model_and_optimizer(hyperparameters, checkpoint_path=None):
    model = TransformerModel(input_dim_lp=34, model_dim=512, num_heads=4, num_layers=4, dropout=0.2, loss_weight_init=hyperparameters.phoneme_loss_weight_init).to(hyperparameters.device)
    optimizer = AdamW(model.parameters(), lr=hyperparameters.learning_rate, weight_decay=hyperparameters.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-7,
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=hyperparameters.device)
        state_dict = checkpoint.get('model_state_dict', {})
        if 'loss_weight' in state_dict and 'loss_weight_raw' not in state_dict:
            try:
                lw = state_dict['loss_weight']
                if not isinstance(lw, torch.Tensor):
                    lw = torch.tensor(lw, dtype=torch.float32)
                lw_clamped = lw.clamp(1e-6, 1.0 - 1e-6)
                state_dict['loss_weight_raw'] = torch.logit(lw_clamped)
                del state_dict['loss_weight']
            except Exception:
                pass
        model.load_state_dict(state_dict, strict=False)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler

def init_mssc_loss_fn(hyperparameters):
    mssc_loss_fn: MultiScaleMelLoss | None = None
    if hyperparameters.mssc_enabled:
        mssc_loss_fn = MultiScaleMelLoss(
            scales=hyperparameters.mssc_scales,
            n_mels=hyperparameters.n_mels,
        ).to(hyperparameters.device)
        print(f"[MSSC] Multi-scale mel loss enabled (scales={hyperparameters.mssc_scales})")
    return mssc_loss_fn

def log_metrics(metrics: tuple, epoch: int, optimizer, prefix: str = "train"):
    # Normalise to a fixed-size 7-tuple so the type checker is satisfied.
    # Pad with 0.0 for any components that aren't present in older-style tuples.
    m = list(metrics)
    while len(m) < 7:
        m.append(0.0)
    (average_epoch_loss, average_phoneme_acc, average_mel_loss, average_phon_loss,
     average_mssc_sc, average_mssc_mag, _std_epoch_loss) = m[:7]

    def _unwrap_metric(x):
        return x.item() if isinstance(x, torch.Tensor) else x

    mlflow.log_metric(f"{prefix}/average_epoch_loss", _unwrap_metric(average_epoch_loss), step=epoch)
    mlflow.log_metric(f"{prefix}/phoneme_accuracy", _unwrap_metric(average_phoneme_acc), step=epoch)
    mlflow.log_metric(f"{prefix}/mel_loss", _unwrap_metric(average_mel_loss), step=epoch)
    mlflow.log_metric(f"{prefix}/phoneme_loss", _unwrap_metric(average_phon_loss), step=epoch)
    if average_mssc_sc is not None:
        mlflow.log_metric(f"{prefix}/mssc_sc_loss", _unwrap_metric(average_mssc_sc), step=epoch)
    if average_mssc_mag is not None:
        mlflow.log_metric(f"{prefix}/mssc_mag_loss", _unwrap_metric(average_mssc_mag), step=epoch)
    if optimizer is not None:
        mlflow.log_metric(f"{prefix}/learning_rate", _unwrap_metric(optimizer.param_groups[0]['lr']), step=epoch)


def _log_source_metrics(hyperparameters, epoch: int) -> None:
    # Log per-source (voiced/unvoiced) train & val metrics to MLflow.
    for prefix, attr in (("train", "last_train_source_metrics"),
                         ("val", "last_val_source_metrics")):
        source_metrics = getattr(hyperparameters, attr, None) or {}
        for src, metrics in source_metrics.items():
            for key, value in metrics.items():
                try:
                    mlflow.log_metric(f"{prefix}/{src}/{key}", float(value), step=epoch)
                except Exception:
                    pass

def run_training_loop(model: TransformerModel, optimizer, scheduler, train_loader, val_loader, test_loader, hyperparameters, mssc_loss_fn, train_on: str = "voiced", plot_every: int = 5, history_csv_path=None):
    initial_lr = optimizer.param_groups[0]['lr']
    early_stopping = {
        "patience": 20,
        "min_delta": 1e-4,
        "best_loss": float('inf'),
        "epochs_no_improve": 0,
        "best_epoch": -1
    }
    early_stop_triggered = False

    history: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_phoneme_acc": [],
        "val_phoneme_acc": [],
        "lr": [],
    }

    for epoch in tqdm(range(hyperparameters.epochs), desc="Training epochs"):
        if hyperparameters.warmup_epochs > 0 and epoch < hyperparameters.warmup_epochs:
            warmup_factor = float(epoch + 1) / float(hyperparameters.warmup_epochs)
            new_lr = initial_lr * warmup_factor
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            mlflow.log_metric("learning_rate_warmup", float(new_lr), step=epoch)

        train_metrics = model.train_epoch(
            optimizer, train_loader, hyperparameters, mssc_loss_fn=mssc_loss_fn, epoch=epoch, train_on=train_on,
        )
        log_metrics(train_metrics, epoch, optimizer, prefix="train")

        val_metrics = model.evaluate(val_loader, hyperparameters, mssc_loss_fn=mssc_loss_fn, epoch=epoch, train_on=train_on)
        log_metrics(val_metrics, epoch, optimizer, prefix="val")
        _log_source_metrics(hyperparameters, epoch)

        # EARLY STOPPING
        try:
            combined_val_loss = float(val_metrics[0])
        except Exception:
            combined_val_loss = val_metrics[0]
        _es_src = getattr(hyperparameters, "early_stop_source", "both")
        _src_val = getattr(hyperparameters, "last_val_source_metrics", {}) or {}
        if _es_src in ("voiced", "unvoiced") and _es_src in _src_val:
            combined_val_loss = float(_src_val[_es_src]["mel_loss"])

        # Record per-epoch history
        history["epoch"].append(epoch)
        history["train_loss"].append(_metric_value(train_metrics[0]))
        history["val_loss"].append(_metric_value(combined_val_loss))
        history["train_phoneme_acc"].append(_metric_value(train_metrics[1]))
        history["val_phoneme_acc"].append(_metric_value(val_metrics[1]))
        history["lr"].append(float(optimizer.param_groups[0]['lr']))

        # Step LR scheduler on validation loss (if scheduler exists)
        try:
            scheduler.step(combined_val_loss)
            # Log updated learning rate after scheduler step
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", float(current_lr), step=epoch)
        except Exception:
            pass

        # Generate plots every plot_every epochs
        if plot_every and plot_every > 0 and (epoch + 1) % plot_every == 0:
            create_plots(model, val_loader, hyperparameters, epoch)

        # Check for improvement
        if combined_val_loss + early_stopping['min_delta'] < early_stopping['best_loss']:
            early_stopping['best_loss'] = combined_val_loss
            early_stopping['epochs_no_improve'] = 0
            early_stopping['best_epoch'] = epoch
        else:
            early_stopping['epochs_no_improve'] += 1
            if early_stopping['epochs_no_improve'] >= early_stopping['patience']:
                print(f"Early stopping triggered at epoch {epoch}. No improvement in validation loss for {early_stopping['patience']} epochs.")
                early_stop_triggered = True
                break

    history["best_epoch"] = early_stopping['best_epoch']
    history["best_val_loss"] = early_stopping['best_loss']

    if early_stop_triggered:
        print(f"Training stopped early at epoch {epoch}. Best validation loss was {early_stopping['best_loss']} at epoch {early_stopping['best_epoch']}.")
    else:
        print(f"Training completed for {hyperparameters.epochs} epochs. Best validation loss was {early_stopping['best_loss']} at epoch {early_stopping['best_epoch']}.")

    if history_csv_path is not None:
        _write_history_csv(history, history_csv_path)

    return history

def _metric_value(x) -> float:
    # Convert a metric (Tensor/float) to a plain float.
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    try:
        return float(x)
    except Exception:
        return 0.0

def _write_history_csv(history: dict, path) -> None:
    # Write a per-epoch training history dict to a local CSV file.
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in ("epoch", "train_loss", "val_loss", "train_phoneme_acc", "val_phoneme_acc", "lr") if k in history]
    rows = zip(*(history.get(k, []) for k in keys))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)

def run_evaluation(model: TransformerModel, test_loader, hyperparameters, mssc_loss_fn):
    metrics = model.evaluate(test_loader, hyperparameters, mssc_loss_fn=mssc_loss_fn, epoch=hyperparameters.epochs, train_on=TRAIN_ON)
    log_metrics(metrics, hyperparameters.epochs, None, prefix="test")

def run_full_metrics_evaluation(
    model: TransformerModel,
    test_loader: DataLoader,
    hyperparameters,
    checkpoint_path: str | None = None,
    max_batches: int | None = None,
    training_history=None,
    train_on: str = "voiced",
):
    # Run the full 6-group metrics suite (specs/metrics.md) on the test set.
    print("\n" + "=" * 60)
    print("Running Full 6-Group Metrics Suite (specs/metrics.md)")
    print("=" * 60)

    report = evaluate_model(
        model=model,
        dataloader=test_loader,
        hyperparameters=hyperparameters,
        phoneme_inventory=PHONEME_INVENTORY,
        checkpoint_path=checkpoint_path,
        max_batches=max_batches,
        log_to_mlflow=True,
        step=hyperparameters.epochs,
        training_history=training_history,
        train_on=train_on,
    )

    # Print summary
    suite = MetricsSuite(
        model=model,
        hyperparameters=hyperparameters,
        phoneme_inventory=PHONEME_INVENTORY,
        checkpoint_path=checkpoint_path,
    )
    print(suite.summary(report))

    # Save the confusion matrix PNG locally
    cm = report.get("intelligibility", {}).get("confusion_matrix")
    if cm is not None:
        try:
            labels = report["intelligibility"].get("confusion_labels", [])
            fig, ax = plt.subplots(figsize=(16, 14))
            im = ax.imshow(cm, aspect='auto', origin='lower', cmap='Blues', interpolation='nearest')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground Truth")
            ax.set_title(f"Phoneme Confusion Matrix ({len(labels)} classes)")
            if labels:
                step = max(1, len(labels) // 40)
                tick_positions = list(range(0, len(labels), step))
                tick_labels = [labels[i] for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=7, rotation=90)
                ax.set_yticklabels(tick_labels, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046)

            save_dir = Path(hyperparameters.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            cm_path = save_dir / "confusion_matrix.png"
            fig.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[metrics] Confusion matrix saved to {cm_path}")
        except Exception as e:
            print(f"[metrics] Could not save confusion matrix: {e}")

    return report

def run_membrane_test(model: TransformerModel, membrane_loader: DataLoader, hyperparameters, checkpoint_path: str, train_on: str = "voiced"):
    # Evaluate a checkpoint on the membrane (new recording technique) test set.
    print("\n" + "=" * 60)
    print("MEMBRANE TEST-SET EVALUATION (new recording technique)")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Input branch    : {train_on}")
    print(f"Membrane set    : {len(membrane_loader.dataset)} recordings / {len(membrane_loader)} batches")  # type: ignore[arg-type]
    print("=" * 60)

    # Full 6-group metrics suite on the membrane test set.
    run_full_metrics_evaluation(
        model=model,
        test_loader=membrane_loader,
        hyperparameters=hyperparameters,
        checkpoint_path=checkpoint_path,
        max_batches=None,
        train_on=train_on,
    )

    # Comparison plots (inputs, output vs GT mel, phoneme timeline).
    print("\n[MEMBRANE] Generating comparison plots for the membrane test set ...")
    create_plots(
        model,
        membrane_loader,
        hyperparameters,
        epoch=0,
        train_on=train_on,
        plot_dir_name="membrane_test",
    )

def create_plots(model: TransformerModel, validation_loader, hyperparameters, epoch: int, train_on: str | None = None, plot_dir_name: str | None = None):
    # Create a comprehensive visualization of model inputs, outputs, and ground truth.
    model.eval()
    phoneme_inventory = PHONEME_INVENTORY
    save_dir = Path(hyperparameters.save_dir)
    if plot_dir_name is not None:
        save_dir = save_dir / plot_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get a single batch from the validation loader
    try:
        batch = next(iter(validation_loader))
    except StopIteration:
        print("[create_plots] Validation loader is empty, skipping.")
        return

    train_on = train_on if train_on is not None else TRAIN_ON
    source_key = 'unvoiced' if train_on == 'both' else train_on  # 'voiced', 'unvoiced', or 'both' → 'voiced'
    lp_seq = batch[source_key]['lp'].to(hyperparameters.device)
    pha_seq = batch[source_key]['pha'].to(hyperparameters.device)
    amp_seq = batch[source_key]['amp'].to(hyperparameters.device)
    dop_seq = batch[source_key]['dop'].to(hyperparameters.device)
    lengths = batch[source_key]['lengths'].to(hyperparameters.device)
    gt_seq = batch['gt']['data'].to(hyperparameters.device)
    gt_lengths = batch['gt']['lengths'].to(hyperparameters.device)
    phoneme_gt = batch['phoneme']['data'].to(hyperparameters.device)
    phoneme_lengths = batch['phoneme']['lengths'].to(hyperparameters.device)

    with torch.no_grad():
        mel_out, out_lengths, phoneme_logits = model(
            lp_seq, pha_seq, amp_seq, dop_seq, lengths=lengths,
        )

    # Move everything to CPU for plotting
    lp_seq_cpu = lp_seq.cpu().numpy()
    dop_seq_cpu = dop_seq.cpu().numpy()
    amp_seq_cpu = amp_seq.cpu().numpy()
    pha_seq_cpu = pha_seq.cpu().numpy()
    mel_out_cpu = mel_out.cpu().numpy()
    gt_seq_cpu = gt_seq.cpu().numpy()
    phoneme_gt_cpu = phoneme_gt.cpu().numpy()
    phoneme_logits_cpu = phoneme_logits.cpu().numpy()
    lengths_cpu = lengths.cpu().numpy()
    out_lengths_cpu = out_lengths.cpu().numpy()
    gt_lengths_cpu = gt_lengths.cpu().numpy()
    phoneme_lengths_cpu = phoneme_lengths.cpu().numpy()

    n_samples = min(4, lp_seq.size(0))

    for i in range(n_samples):
        T_in = int(lengths_cpu[i])
        T_out = int(out_lengths_cpu[i])
        T_gt = int(gt_lengths_cpu[i])
        T_phon = int(phoneme_lengths_cpu[i])

        if T_out == 0 or T_gt == 0:
            continue

        # Clip to the shorter of output and GT for comparison
        T_vis = min(T_out, T_gt)

        lp_slice = lp_seq_cpu[i, :T_in, :]
        dop_slice = dop_seq_cpu[i, :T_in, :].T
        amp_slice = amp_seq_cpu[i, :T_in, 0]
        pha_slice = pha_seq_cpu[i, :T_in, 0]
        mel_pred = mel_out_cpu[i, :T_vis, :].T
        mel_gt = gt_seq_cpu[i, :T_vis, :].T
        err_map = np.abs(mel_gt - mel_pred)

        # Phoneme prediction: argmax over logits
        phon_pred = phoneme_logits_cpu[i, :T_out, :].argmax(axis=-1)
        phon_gtruth = phoneme_gt_cpu[i, :T_phon]

        # --- Build the figure ---
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 2, 1.5], hspace=0.35, wspace=0.3)

        # Row 0: LP Input + Doppler Spectrogram
        ax_lp = fig.add_subplot(gs[0, 0])
        im_lp = ax_lp.imshow(lp_slice.T, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
        ax_lp.set_title(f'LP Residual Features [{lp_slice.shape[-1]} dim]')
        ax_lp.set_ylabel('Feature dim')
        plt.colorbar(im_lp, ax=ax_lp, fraction=0.05)

        ax_dop = fig.add_subplot(gs[0, 1])
        # Downsample Doppler for display if too many bins
        disp_dop = dop_slice[::4, :] if dop_slice.shape[0] > 256 else dop_slice
        im_dop = ax_dop.imshow(disp_dop, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
        ax_dop.set_title(f'UltraSR Doppler [{dop_slice.shape[0]} → {disp_dop.shape[0]} bins]')
        ax_dop.set_ylabel('Doppler bin (decimated)')
        plt.colorbar(im_dop, ax=ax_dop, fraction=0.05)

        # Row 0 col 2: combined UltraSR gradient line plot
        ax_grad = fig.add_subplot(gs[0, 2])
        t_axis = np.arange(T_in)
        ax_grad.plot(t_axis, amp_slice, color='steelblue', label='Amp', alpha=0.8, linewidth=0.8)
        ax_grad.plot(t_axis, pha_slice, color='darkorange', label='Pha', alpha=0.8, linewidth=0.8)
        ax_grad.set_title('UltraSR Gradients')
        ax_grad.set_xlabel('Time frame')
        ax_grad.legend(fontsize=8)
        ax_grad.set_ylabel('Amplitude')

        # Row 1: Reconstructed vs Ground Truth Mel
        ax_pred = fig.add_subplot(gs[1, 0])
        im_pred = ax_pred.imshow(mel_pred, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
        ax_pred.set_title(f'Reconstructed Mel (output) [{T_vis} frames]')
        ax_pred.set_ylabel('Mel bin')
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.05)

        ax_gt = fig.add_subplot(gs[1, 1])
        im_gt = ax_gt.imshow(mel_gt, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
        ax_gt.set_title(f'Ground Truth Mel [{T_vis} frames]')
        ax_gt.set_ylabel('Mel bin')
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.05)

        # Row 1 col 2: Error map
        ax_err = fig.add_subplot(gs[1, 2])
        vmax = float(np.percentile(err_map, 98))
        im_err = ax_err.imshow(err_map, aspect='auto', origin='lower', cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
        ax_err.set_title(f'Error |GT - Pred| (p98={vmax:.3f})')
        ax_err.set_ylabel('Mel bin')
        plt.colorbar(im_err, ax=ax_err, fraction=0.05)

        # Row 2: Time-aligned comparison — overlay output & GT
        ax_compare = fig.add_subplot(gs[2, :])
        # Show a few representative mel bins as line plots over time
        mel_bins_to_plot = [20, 40, 60, 80, 100]
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(mel_bins_to_plot)))
        t_vis_axis = np.arange(T_vis)
        for idx, b in enumerate(mel_bins_to_plot):
            if b < mel_pred.shape[0]:
                ax_compare.plot(t_vis_axis, mel_pred[b], color=colors[idx], linestyle='-',
                                label=f'Pred mel[{b}]', alpha=0.8, linewidth=0.8)
                ax_compare.plot(t_vis_axis, mel_gt[b], color=colors[idx], linestyle='--',
                                label=f'GT mel[{b}]', alpha=0.5, linewidth=0.8)
        ax_compare.set_title('Selected Mel Bins: Predicted (solid) vs Ground Truth (dashed)')
        ax_compare.set_xlabel('Time frame')
        ax_compare.set_ylabel('Magnitude')
        ax_compare.legend(fontsize=7, ncol=2)

        # Row 3: Phoneme Timeline — predicted vs ground truth
        ax_phon = fig.add_subplot(gs[3, :])
        cmap_phon = plt.get_cmap('tab20', max(2, len(phoneme_inventory)))
        n_phon = len(phoneme_inventory)

        # Build segments for predicted phonemes
        def _phoneme_segments(seq, max_len):
            segs = []
            if len(seq) == 0:
                return segs
            cur_label = int(seq[0])
            start = 0
            for t in range(1, min(len(seq), max_len)):
                lbl = int(seq[t])
                if lbl != cur_label:
                    segs.append((start, t, cur_label))
                    start = t
                    cur_label = lbl
            segs.append((start, min(len(seq), max_len), cur_label))
            return segs

        pred_segs = _phoneme_segments(phon_pred, T_vis)
        gt_segs = _phoneme_segments(phon_gtruth, T_vis)

        ax_phon.set_ylim(0, 2)
        ax_phon.set_yticks([0.5, 1.5])
        ax_phon.set_yticklabels(['Prediction', 'Ground Truth'], fontsize=9)
        ax_phon.set_xlim(0, T_vis)
        ax_phon.set_xlabel('Time frame')
        ax_phon.set_title('Phoneme Timeline: Predicted vs Ground Truth')

        for segs, y_offset in [(pred_segs, 1.0), (gt_segs, 0.0)]:
            for start, end, label in segs:
                color = cmap_phon(label % cmap_phon.N)
                rect = Rectangle((start, y_offset), end - start, 1, facecolor=color, edgecolor='none', alpha=0.75)
                ax_phon.add_patch(rect)
                if end - start >= max(6, T_vis // 40):
                    label_text = phoneme_inventory[label] if 0 <= label < n_phon else '?'
                    ax_phon.text((start + end) / 2, y_offset + 0.5, label_text,
                                 ha='center', va='center', fontsize=7, color='black', clip_on=True)

        # Add vertical gridlines at phoneme boundaries
        all_boundaries = set()
        for segs in [pred_segs, gt_segs]:
            for _, end, _ in segs:
                all_boundaries.add(end)
        for b in sorted(all_boundaries):
            ax_phon.axvline(b, color='gray', alpha=0.2, linewidth=0.4)

        fig.suptitle(f'Epoch {epoch:04d} — Sample {i} | LP dim={lp_slice.shape[-1]} | '
                     f'Mel bins={mel_pred.shape[0]} | Phonemes={n_phon}',
                     fontsize=13, fontweight='bold')

        save_path = save_dir / f'epoch_{epoch:04d}_sample_{i}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[create_plots] Saved {save_path}')

    model.train()

def save_final_model(model: TransformerModel, optimizer, hyperparameters, scheduler):
    final_checkpoint_path = Path(hyperparameters.save_dir) / f"model_final"
    save_checkpoint(model, optimizer, hyperparameters.epochs, None, final_checkpoint_path, scheduler=scheduler)
    print(f"Final model saved to {final_checkpoint_path}")

if __name__ == "__main__":
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.download_data and not args.membrane_test:
        print('[INFO] Downloading data...')
        DATA_PATH = Path(args.data_path)
        UNVOICED_PATH = DATA_PATH / "unvoiced_samples"
        VOICED_PATH = DATA_PATH / "voiced_samples"
        TEXTGRID_PATH = DATA_PATH / "textgrids"
        download_data(args.backend_url, DATA_PATH, VOICED_PATH, UNVOICED_PATH, args.debug, TEXTGRID_PATH)

    if args.add_membrane_to_training:
        data_path = Path(args.data_path)
        voiced_p = data_path / "voiced_samples"
        unvoiced_p = data_path / "unvoiced_samples"
        textgrid_p = data_path / "textgrids"
        print(f'[MEMBRANE-TRAINING] Downloading {len(MEMBRANE_RECORDING_UUIDS)} membrane recordings into main dataset directories ...')
        download_data_for_uuids(
            args.backend_url,
            MEMBRANE_RECORDING_UUIDS,
            voiced_path=voiced_p,
            unvoiced_path=unvoiced_p,
            textgrid_path=textgrid_p,
            debug=args.debug,
        )
        print(f'[MEMBRANE-TRAINING] Done. Membrane recordings will be included in the train/val/test split.')

    DEBUG = args.debug
    TRAIN_ON = args.train_on
    
    hyperparameters = Hyperparameters()
    hyperparameters.epochs = args.epochs
    hyperparameters.save_dir = Path(hyperparameters.save_dir) / args.run_name
    hyperparameters.save_dir.mkdir(parents=True, exist_ok=True)
    hyperparameters.batch_size = args.batch_size
    TRAIN_ON = args.train_on

    setup_manual_seed(hyperparameters.seed)
    if args.cuda_device is not None:
        hyperparameters.device = f"cuda:{args.cuda_device}"
    hyperparameters.device = normalize_device_string(hyperparameters.device)
    hyperparameters.device = setup_torch(hyperparameters.device)

    # Log selected device and CUDA visibility for debug and reproducibility.
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if isinstance(hyperparameters.device, str) and hyperparameters.device.startswith('cuda') and torch.cuda.is_available():
        try:
            current_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_index)
        except Exception:
            current_index = None
            device_name = None
        print(f"[DEVICE] CUDA_VISIBLE_DEVICES={cuda_visible_devices}; requested_device={args.cuda_device if args.cuda_device is not None else 'default'}; resolved_device={hyperparameters.device}; current_device={current_index}; device_name={device_name}")
    else:
        print(f"[DEVICE] CUDA_VISIBLE_DEVICES={cuda_visible_devices}; resolved_device={hyperparameters.device}")

    experiment = setup_mlflow(MLFLOW_SERVER_URL, args.experiment_group)

    mlflow.start_run(run_name=args.run_name, experiment_id=experiment.experiment_id)

    if args.membrane_test:
        if not args.checkpoint_path or not Path(args.checkpoint_path).exists():
            parser.error("--membrane-test requires an existing --checkpoint-path (the model to evaluate).")

        membrane_data_path = Path(args.membrane_data_path)
        membrane_voiced_path = Path(args.membrane_voiced_path) if args.membrane_voiced_path else membrane_data_path / "voiced_samples"
        membrane_unvoiced_path = Path(args.membrane_unvoiced_path) if args.membrane_unvoiced_path else membrane_data_path / "unvoiced_samples"
        membrane_textgrid_path = Path(args.membrane_textgrid_path) if args.membrane_textgrid_path else membrane_data_path / "textgrids"

        if args.download_data:
            print(f'[INFO] Downloading membrane test recordings into {membrane_data_path} ...')
            download_data_for_uuids(
                args.backend_url,
                MEMBRANE_RECORDING_UUIDS,
                voiced_path=membrane_voiced_path,
                unvoiced_path=membrane_unvoiced_path,
                textgrid_path=membrane_textgrid_path,
                debug=args.debug,
            )

        try:
            mlflow.set_tag("test_set", "membrane")
            mlflow.set_tag("checkpoint_path", args.checkpoint_path)
        except Exception:
            pass

        print(f'[MEMBRANE] Loading model from {args.checkpoint_path}')
        model, _, _ = init_model_and_optimizer(hyperparameters, checkpoint_path=args.checkpoint_path)

        membrane_loader = load_membrane_test_loader(
            hyperparameters=hyperparameters,
            voiced_path=membrane_voiced_path,
            unvoiced_path=membrane_unvoiced_path,
            textgrid_path=membrane_textgrid_path,
            uuids=MEMBRANE_RECORDING_UUIDS,
            debug=args.debug,
        )

        run_membrane_test(model, membrane_loader, hyperparameters, args.checkpoint_path, train_on=TRAIN_ON)

        mlflow.end_run()
        sys.exit(0)

    # Initializing Model, Optimizer, Scheduler and Loss function
    if DEBUG: print('Init Model')
    model, optimizer, scheduler = init_model_and_optimizer(hyperparameters, checkpoint_path=args.checkpoint_path if args.start_from_checkpoint else None)
    mssc_loss_fn = init_mssc_loss_fn(hyperparameters)

    # Initializing and Loading Datasets
    if DEBUG: print('Loading Dataset')
    train_loader, val_loader, test_loader = load_datasets(
        voiced_path=Path(args.voiced_path),
        unvoiced_path=Path(args.unvoiced_path),
        textgrid_path=Path(args.textgrid_path),
        hyperparameters=hyperparameters,
        debug=args.debug
    )
    
    # Run Training Loop (skip if --metrics-only)
    if not args.metrics_only:
        if DEBUG: print('Starting Training Loop')
        run_training_loop(model, optimizer, scheduler, train_loader, val_loader, test_loader, hyperparameters, mssc_loss_fn)
        # Run basic evaluation
        if DEBUG: print('Running Eval')
        run_evaluation(model, test_loader, hyperparameters, mssc_loss_fn)

        # Save Results
        create_plots(model, val_loader, hyperparameters, epoch=hyperparameters.epochs - 1)
        save_final_model(model, optimizer, hyperparameters, scheduler)

    # --- Full 6-Group Metrics Suite (specs/metrics.md) ---
    if args.full_metrics or args.metrics_only:
        if DEBUG: print('Running Full Metrics Suite')
        checkpoint_for_size = args.checkpoint_path if args.start_from_checkpoint else None
        run_full_metrics_evaluation(
            model=model,
            test_loader=test_loader,
            hyperparameters=hyperparameters,
            checkpoint_path=checkpoint_for_size,
            max_batches=args.metrics_max_batches,
            train_on=TRAIN_ON,
        )

    mlflow.end_run()    