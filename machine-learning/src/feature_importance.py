from __future__ import annotations

import argparse
import csv
import sys
import warnings
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from torch import Tensor, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from dataset.transformer_dataset import SSIDataset, pad_collate_fn
from dataset.residual_features import ResidualFeatureExtractor
from models.transformer import TransformerModel
from utils.hyperparameters_transformer import Hyperparameters
from utils.utils_transformer import PHONEME_INVENTORY

_RESIDUAL_EXTRACTOR = ResidualFeatureExtractor()
_LP_FEATURE_NAMES = _RESIDUAL_EXTRACTOR.get_feature_names()  # 34 names

FEATURE_GROUPS: List[Dict] = [
    {
        "name": "Subband Energy Envelopes",
        "short_name": "subband_energy",
        "group_id": 1,
        "stream": "lp",
        "indices": list(range(0, 14)),
        "channel_names": _LP_FEATURE_NAMES[0:14],
    },
    {
        "name": "Spectral Flux",
        "short_name": "spectral_flux",
        "group_id": 2,
        "stream": "lp",
        "indices": [14],
        "channel_names": [_LP_FEATURE_NAMES[14]],
    },
    {
        "name": "Spectral Moments",
        "short_name": "spectral_moments",
        "group_id": 3,
        "stream": "lp",
        "indices": [15, 16, 17],
        "channel_names": _LP_FEATURE_NAMES[15:18],
    },
    {
        "name": "Zero Crossing Rate",
        "short_name": "zcr",
        "group_id": 4,
        "stream": "lp",
        "indices": [18],
        "channel_names": [_LP_FEATURE_NAMES[18]],
    },
    {
        "name": "Spectral Flatness",
        "short_name": "spectral_flatness",
        "group_id": 5,
        "stream": "lp",
        "indices": [19],
        "channel_names": [_LP_FEATURE_NAMES[19]],
    },
    {
        "name": "MFCCs",
        "short_name": "mfcc",
        "group_id": 6,
        "stream": "lp",
        "indices": list(range(20, 33)),
        "channel_names": _LP_FEATURE_NAMES[20:33],
    },
    {
        "name": "Teager Energy Operator",
        "short_name": "teo",
        "group_id": 7,
        "stream": "lp",
        "indices": [33],
        "channel_names": [_LP_FEATURE_NAMES[33]],
    },
    {
        "name": "Phase Gradient (UltraSR)",
        "short_name": "phase_grad",
        "group_id": 8,
        "stream": "pha",
        "indices": [0],
        "channel_names": ["phase_gradient"],
    },
    {
        "name": "Amplitude Gradient (UltraSR)",
        "short_name": "amp_grad",
        "group_id": 9,
        "stream": "amp",
        "indices": [0],
        "channel_names": ["amplitude_gradient"],
    },
    {
        "name": "Doppler Spectrogram (UltraSR)",
        "short_name": "doppler",
        "group_id": 10,
        "stream": "dop",
        "indices": list(range(0, 1024)),
        "channel_names": [f"doppler_bin_{i}" for i in range(1024)],
    },
]

N_GROUPS = len(FEATURE_GROUPS)  # 10

def _build_doppler_blocks(block_size: int) -> List[Tuple[int, int]]:
    """Split the 1024 Doppler bins into blocks for permutation importance."""
    blocks = []
    for start in range(0, 1024, block_size):
        end = min(start + block_size, 1024)
        blocks.append((start, end))
    return blocks

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Feature importance analysis for SSI Transformer model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to the trained model checkpoint (.pth).")
    p.add_argument("--data-dir", type=str, default="transformer_data",
                   help="Base directory for the dataset.")
    p.add_argument("--voiced-path", type=str, default=None,
                   help="Voiced samples directory (default: <data-dir>/voiced_samples).")
    p.add_argument("--unvoiced-path", type=str, default=None,
                   help="Unvoiced samples directory (default: <data-dir>/unvoiced_samples).")
    p.add_argument("--textgrid-path", type=str, default=None,
                   help="TextGrid directory (default: <data-dir>/textgrids).")
    p.add_argument("--method", type=str, default="all",
                   choices=["ablation", "permutation", "attention", "all"],
                   help="Which analysis method(s) to run.")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Number of test samples to evaluate (default: 50).")
    p.add_argument("--output-dir", type=str, default="feature_importance_results",
                   help="Directory for CSV exports and plots.")
    p.add_argument("--device", type=str, default=None,
                   help="Device override (e.g. 'cuda:0', 'cpu').")
    p.add_argument("--train-on", type=str, default="voiced",
                   choices=["voiced", "unvoiced", "both"],
                   help="Input branch to use (default: voiced).")
    p.add_argument("--doppler-block-size", type=int, default=64,
                   help="Block size for Doppler permutation (default: 64 → 16 blocks).")
    p.add_argument("--doppler-subsplit", type=int, default=0,
                   help="If >0, split Doppler into N sub-groups for ablation "
                        "(e.g. 16 = 16 frequency bands).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    return p

def load_model(checkpoint_path: str, device: str) -> TransformerModel:
    """Load a trained TransformerModel from a checkpoint."""
    hp = Hyperparameters()
    hp.device = device

    model = TransformerModel(
        input_dim_lp=34,
        model_dim=512,
        num_heads=4,
        num_layers=4,
        dropout=0.2,
        loss_weight_init=hp.phoneme_loss_weight_init,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Handle legacy loss_weight → loss_weight_raw migration
    if "loss_weight" in state_dict and "loss_weight_raw" not in state_dict:
        try:
            lw = state_dict["loss_weight"]
            if not isinstance(lw, Tensor):
                lw = torch.tensor(lw, dtype=torch.float32)
            lw_clamped = lw.clamp(1e-6, 1.0 - 1e-6)
            state_dict["loss_weight_raw"] = torch.logit(lw_clamped)
            del state_dict["loss_weight"]
        except Exception:
            pass

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[load] Model loaded from {checkpoint_path} ({sum(p.numel() for p in model.parameters()):,} params)")
    return model

def load_test_loader(
    data_dir: str,
    voiced_path_str: Optional[str],
    unvoiced_path_str: Optional[str],
    textgrid_path_str: Optional[str],
    n_samples: int,
    device: str,
    seed: int,
) -> DataLoader:
    """Load test-split data via SSIDataset, limited to n_samples."""
    data_dir_p = Path(data_dir)
    voiced = Path(voiced_path_str) if voiced_path_str else data_dir_p / "voiced_samples"
    unvoiced = Path(unvoiced_path_str) if unvoiced_path_str else data_dir_p / "unvoiced_samples"
    textgrids = Path(textgrid_path_str) if textgrid_path_str else data_dir_p / "textgrids"

    hp = Hyperparameters()
    hp.device = device
    hp.seed = seed

    dataset = SSIDataset(hp, split="test", debug=False)
    dataset._load_data(voiced, unvoiced, textgrids)

    actual_n = min(n_samples, len(dataset))
    print(f"[data] Test set: {len(dataset)} samples, using {actual_n} for analysis.")

    # Deterministic sub-sampling
    indices = list(range(len(dataset)))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    indices = indices[:actual_n]

    loader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(indices)
    )

    return loader

def _forward_model(
    model: TransformerModel,
    lp: Tensor,
    pha: Tensor,
    amp: Tensor,
    dop: Tensor,
    lengths: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run model forward, return (mel_out, out_lengths, phoneme_logits)."""
    with no_grad():
        mel_out, out_lengths, phon_logits = model(lp, pha, amp, dop, lengths=lengths)
    return mel_out, out_lengths, phon_logits

def _compute_mse(pred_mel: Tensor, gt_mel: Tensor, valid_len: int) -> float:
    """Mean squared error over valid (non-padded) time frames."""
    p = pred_mel[:valid_len]
    g = gt_mel[:valid_len]
    return float(torch.nn.functional.mse_loss(p, g).item())

def _compute_lsd(pred_mel: Tensor, gt_mel: Tensor, valid_len: int) -> float:
    """Log-Spectral Distance (dB) over valid frames."""
    p = pred_mel[:valid_len].detach().cpu().numpy()
    g = gt_mel[:valid_len].detach().cpu().numpy()
    diff = p - g  # log-mel domain difference
    lsd_per_frame = np.sqrt(np.mean(diff ** 2, axis=1))
    return float(np.mean(lsd_per_frame))

def run_ablation(
    model: TransformerModel,
    test_loader: DataLoader,
    device: str,
    train_on: str,
    doppler_subsplits: int = 0,
) -> Dict:
    """Zero out each feature group, measure loss degradation."""
    print("\n" + "=" * 60)
    print("METHOD 1: Feature Group Ablation")
    print("=" * 60)

    # Build the list of groups to ablate. If doppler_subsplits > 0,
    # replace the single Doppler group with N sub-groups.
    groups_to_ablate: List[Dict] = []
    for g in FEATURE_GROUPS:
        if g["stream"] == "dop" and doppler_subsplits > 0:
            # Split Doppler into N frequency sub-bands
            bins_per_split = 1024 // doppler_subsplits
            for s in range(doppler_subsplits):
                start = s * bins_per_split
                end = 1024 if s == doppler_subsplits - 1 else (s + 1) * bins_per_split
                groups_to_ablate.append({
                    "name": f"Doppler band {start}-{end} Hz",
                    "short_name": f"dop_{start}_{end}",
                    "group_id": f"dop_split_{s}",
                    "stream": "dop",
                    "indices": list(range(start, end)),
                    "channel_names": [f"doppler_bin_{i}" for i in range(start, end)],
                })
        else:
            groups_to_ablate.append(g)

    results: Dict[str, List[float]] = {
        "group_name": [],
        "short_name": [],
        "group_id": [],
        "stream": [],
        "n_channels": [],
        "mse_delta_mean": [],
        "mse_delta_std": [],
        "lsd_delta_mean": [],
        "lsd_delta_std": [],
    }

    # ── Baseline pass ──
    baseline_mse: List[float] = []
    baseline_lsd: List[float] = []
    source_key = "unvoiced" if train_on == "unvoiced" else "voiced"

    print("Running baseline (all features) ...")
    for batch in tqdm(test_loader, desc="Baseline", leave=False):
        lp = batch[source_key]["lp"].to(device)
        pha = batch[source_key]["pha"].to(device)
        amp = batch[source_key]["amp"].to(device)
        dop = batch[source_key]["dop"].to(device)
        lengths = batch[source_key]["lengths"].to(device)
        gt = batch["gt"]["data"].to(device)
        gt_lengths = batch["gt"]["lengths"].to(device)

        mel_out, out_lens, _ = _forward_model(model, lp, pha, amp, dop, lengths)

        for i in range(lp.size(0)):
            vlen = min(int(out_lens[i].item()), int(gt_lengths[i].item()))
            if vlen == 0:
                continue
            baseline_mse.append(_compute_mse(mel_out[i], gt[i], vlen))
            baseline_lsd.append(_compute_lsd(mel_out[i], gt[i], vlen))

    mean_baseline_mse = float(np.mean(baseline_mse))
    mean_baseline_lsd = float(np.mean(baseline_lsd))
    print(f"  Baseline MSE={mean_baseline_mse:.6f}, LSD={mean_baseline_lsd:.4f} dB")

    # ── Ablation per group ──
    for group in tqdm(groups_to_ablate, desc="Ablating groups"):
        grp_mse_deltas: List[float] = []
        grp_lsd_deltas: List[float] = []

        for batch in test_loader:
            lp = batch[source_key]["lp"].clone().to(device)
            pha = batch[source_key]["pha"].clone().to(device)
            amp = batch[source_key]["amp"].clone().to(device)
            dop = batch[source_key]["dop"].clone().to(device)
            lengths = batch[source_key]["lengths"].to(device)
            gt = batch["gt"]["data"].to(device)
            gt_lengths = batch["gt"]["lengths"].to(device)

            # Zero the target group's channels
            stream = group["stream"]
            idxs = group["indices"]
            if stream == "lp":
                lp[:, :, idxs] = 0.0
            elif stream == "pha":
                pha[:, :, idxs] = 0.0
            elif stream == "amp":
                amp[:, :, idxs] = 0.0
            elif stream == "dop":
                dop[:, :, idxs] = 0.0

            mel_out, out_lens, _ = _forward_model(model, lp, pha, amp, dop, lengths)

            for i in range(lp.size(0)):
                vlen = min(int(out_lens[i].item()), int(gt_lengths[i].item()))
                if vlen == 0:
                    continue
                abl_mse = _compute_mse(mel_out[i], gt[i], vlen)
                abl_lsd = _compute_lsd(mel_out[i], gt[i], vlen)
                grp_mse_deltas.append(abl_mse - baseline_mse[i] if i < len(baseline_mse) else 0.0)
                grp_lsd_deltas.append(abl_lsd - baseline_lsd[i] if i < len(baseline_lsd) else 0.0)

        results["group_name"].append(group["name"])
        results["short_name"].append(group["short_name"])
        results["group_id"].append(str(group["group_id"]))
        results["stream"].append(group["stream"])
        results["n_channels"].append(len(group["indices"]))
        results["mse_delta_mean"].append(float(np.mean(grp_mse_deltas)))
        results["mse_delta_std"].append(float(np.std(grp_mse_deltas)))
        results["lsd_delta_mean"].append(float(np.mean(grp_lsd_deltas)))
        results["lsd_delta_std"].append(float(np.std(grp_lsd_deltas)))

    return results

def run_permutation(
    model: TransformerModel,
    test_loader: DataLoader,
    device: str,
    train_on: str,
    doppler_block_size: int = 64,
) -> Dict:
    """Permute each channel's temporal order, measure degradation."""
    print("\n" + "=" * 60)
    print("METHOD 2: Per-Channel Permutation Importance")
    print("=" * 60)

    source_key = "unvoiced" if train_on == "unvoiced" else "voiced"

    # ── Baseline pass (reuse same compute, but simpler to re-run) ──
    baseline_mse: List[float] = []
    baseline_lsd: List[float] = []

    for batch in tqdm(test_loader, desc="Baseline", leave=False):
        lp = batch[source_key]["lp"].to(device)
        pha = batch[source_key]["pha"].to(device)
        amp = batch[source_key]["amp"].to(device)
        dop = batch[source_key]["dop"].to(device)
        lengths = batch[source_key]["lengths"].to(device)
        gt = batch["gt"]["data"].to(device)
        gt_lengths = batch["gt"]["lengths"].to(device)

        mel_out, out_lens, _ = _forward_model(model, lp, pha, amp, dop, lengths)

        for i in range(lp.size(0)):
            vlen = min(int(out_lens[i].item()), int(gt_lengths[i].item()))
            if vlen == 0:
                continue
            baseline_mse.append(_compute_mse(mel_out[i], gt[i], vlen))
            baseline_lsd.append(_compute_lsd(mel_out[i], gt[i], vlen))

    # ── Build per-channel lists ──
    # For LP features: 34 individual channels
    # For pha/amp: 1 each
    # For Doppler: block_size blocks
    channel_list: List[Dict] = []

    for g in FEATURE_GROUPS:
        stream = g["stream"]
        if stream == "dop":
            blocks = _build_doppler_blocks(doppler_block_size)
            for b_start, b_end in blocks:
                channel_list.append({
                    "group_name": g["name"],
                    "short_name": g["short_name"],
                    "stream": stream,
                    "channel_name": f"dop_bins_{b_start}_{b_end}",
                    "indices": list(range(b_start, b_end)),
                })
        else:
            for idx, ch_name in zip(g["indices"], g["channel_names"]):
                channel_list.append({
                    "group_name": g["name"],
                    "short_name": g["short_name"],
                    "stream": stream,
                    "channel_name": ch_name,
                    "indices": [idx],
                })

    print(f"  Permuting {len(channel_list)} channels/blocks across {N_GROUPS} groups...")

    results: Dict[str, List] = {
        "group_name": [],
        "short_name": [],
        "stream": [],
        "channel_name": [],
        "mse_delta_mean": [],
        "mse_delta_std": [],
        "lsd_delta_mean": [],
        "lsd_delta_std": [],
    }

    for ch in tqdm(channel_list, desc="Permuting channels"):
        ch_mse_deltas: List[float] = []
        ch_lsd_deltas: List[float] = []

        for batch in test_loader:
            lp = batch[source_key]["lp"].clone().to(device)
            pha = batch[source_key]["pha"].clone().to(device)
            amp = batch[source_key]["amp"].clone().to(device)
            dop = batch[source_key]["dop"].clone().to(device)
            lengths = batch[source_key]["lengths"].to(device)
            gt = batch["gt"]["data"].to(device)
            gt_lengths = batch["gt"]["lengths"].to(device)

            # Permute the target channel(s) along the time axis
            stream = ch["stream"]
            idxs = ch["indices"]
            tensor_map = {"lp": lp, "pha": pha, "amp": amp, "dop": dop}
            tgt = tensor_map[stream]

            for b in range(tgt.size(0)):
                valid_t = int(lengths[b].item())
                if valid_t <= 1:
                    continue
                for idx in idxs:
                    perm = torch.randperm(valid_t, device=device)
                    tgt[b, :valid_t, idx] = tgt[b, perm, idx]

            mel_out, out_lens, _ = _forward_model(model, lp, pha, amp, dop, lengths)

            for i in range(lp.size(0)):
                vlen = min(int(out_lens[i].item()), int(gt_lengths[i].item()))
                if vlen == 0:
                    continue
                p_mse = _compute_mse(mel_out[i], gt[i], vlen)
                p_lsd = _compute_lsd(mel_out[i], gt[i], vlen)
                ch_mse_deltas.append(p_mse - baseline_mse[i] if i < len(baseline_mse) else 0.0)
                ch_lsd_deltas.append(p_lsd - baseline_lsd[i] if i < len(baseline_lsd) else 0.0)

        results["group_name"].append(ch["group_name"])
        results["short_name"].append(ch["short_name"])
        results["stream"].append(ch["stream"])
        results["channel_name"].append(ch["channel_name"])
        results["mse_delta_mean"].append(float(np.mean(ch_mse_deltas)))
        results["mse_delta_std"].append(float(np.std(ch_mse_deltas)))
        results["lsd_delta_mean"].append(float(np.mean(ch_lsd_deltas)))
        results["lsd_delta_std"].append(float(np.std(ch_lsd_deltas)))

    return results

def run_attention_analysis(
    model: TransformerModel,
    test_loader: DataLoader,
    device: str,
    train_on: str,
) -> Dict:
    """Extract embedding stream magnitudes and self-attention weights."""
    print("\n" + "=" * 60)
    print("METHOD 3: Attention Map & Stream Activation Analysis")
    print("=" * 60)

    source_key = "unvoiced" if train_on == "unvoiced" else "voiced"

    # Storage for activation norms
    stream_l2: Dict[str, List[float]] = {"lp": [], "pha": [], "amp": [], "dop": []}
    attention_maps: List[np.ndarray] = []  # each: [n_layers, T, T]

    embedding_outputs: Dict[str, Tensor] = {}

    def _make_hook(name: str):
        def hook(_module, _input, output):
            embedding_outputs[name] = output.detach()
        return hook

    handles = []
    handles.append(model.pha_embed.register_forward_hook(_make_hook("pha")))
    handles.append(model.amp_embed.register_forward_hook(_make_hook("amp")))
    handles.append(model.dop_embed.register_forward_hook(_make_hook("dop")))
    handles.append(model.lp_embed.register_forward_hook(_make_hook("lp")))

    try:
        sample_count = 0
        max_attention_samples = 4

        for batch in test_loader:
            lp = batch[source_key]["lp"].to(device)
            pha = batch[source_key]["pha"].to(device)
            amp = batch[source_key]["amp"].to(device)
            dop = batch[source_key]["dop"].to(device)
            lengths = batch[source_key]["lengths"].to(device)

            with no_grad():
                _ = model(lp, pha, amp, dop, lengths=lengths)

            # Record embedding stream per-channel L2 norms
            for stream_name in ["pha", "amp", "dop", "lp"]:
                if stream_name in embedding_outputs:
                    act = embedding_outputs[stream_name]  # [B, C, T]
                    n_channels = act.size(1)
                    for b in range(act.size(0)):
                        valid_t = int(lengths[b].item())
                        if valid_t == 0:
                            continue
                        # Per-channel L2 norm: [C]
                        ch_l2 = torch.norm(act[b, :, :valid_t], dim=1)  # [C]
                        # Average per-channel norm
                        avg_per_ch = float(ch_l2.mean().item())
                        stream_l2[stream_name].append(avg_per_ch)

            # Extract attention maps for a few samples (separate encoder pass)
            if sample_count < max_attention_samples:
                for i in range(min(1, lp.size(0))):  # one per batch
                    with no_grad():
                        # Re-run model internals to get attention
                        # We'll do this by replicating forward() up to the encoder
                        att_maps = _extract_attention(model, lp[i:i+1], pha[i:i+1],
                                                      amp[i:i+1], dop[i:i+1],
                                                      lengths[i:i+1])
                        if att_maps is not None:
                            attention_maps.append(att_maps)
                            sample_count += 1
                            if sample_count >= max_attention_samples:
                                break
            if sample_count >= max_attention_samples and len(stream_l2["lp"]) > 100:
                break

    finally:
        for h in handles:
            h.remove()

    # Aggregate stream L2 norms
    stream_stats = {}
    for s in ["lp", "pha", "amp", "dop"]:
        vals = stream_l2[s]
        if vals:
            stream_stats[s] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
        else:
            stream_stats[s] = {"mean": 0.0, "std": 0.0, "n": 0}

    # Compute relative contributions (normalize so they sum to 1)
    total = sum(stream_stats[s]["mean"] for s in stream_stats)
    for s in stream_stats:
        stream_stats[s]["relative"] = stream_stats[s]["mean"] / total if total > 0 else 0.0

    print("  Stream activation (L2 norm, relative contribution):")
    for s in ["lp", "pha", "amp", "dop"]:
        print(f"    {s:6s}: mean={stream_stats[s]['mean']:.3f} "
              f"({stream_stats[s]['relative']*100:.1f}%)")

    return {
        "stream_stats": stream_stats,
        "attention_maps": attention_maps,
        "n_attention_samples": len(attention_maps),
    }

def _extract_attention(
    model: TransformerModel,
    lp: Tensor,
    pha: Tensor,
    amp: Tensor,
    dop: Tensor,
    lengths: Tensor,
) -> Optional[np.ndarray]:
    """Run a separate forward pass that extracts self-attention weights."""
    try:
        if pha.dim() == 2:
            pha = pha.unsqueeze(-1)
        if amp.dim() == 2:
            amp = amp.unsqueeze(-1)

        mask = torch.arange(pha.size(1), device=pha.device).unsqueeze(0) >= lengths.unsqueeze(1)

        pha_feat = model.pha_embed(pha.permute(0, 2, 1))
        amp_feat = model.amp_embed(amp.permute(0, 2, 1))
        dop_feat = model.dop_embed(dop.permute(0, 2, 1))
        lp_feat = model.lp_embed(lp.permute(0, 2, 1))

        fused = torch.cat([pha_feat, amp_feat, dop_feat, lp_feat], dim=1)

        for d_conv in model.dilated_convs:
            fused = torch.relu(d_conv(fused))

        fused = fused.permute(0, 2, 1)
        fused = model.positional_encoding(fused)

        # Manually run transformer encoder layers to collect attention weights
        all_attn = []
        x = fused
        for layer in model.transformer_encoder.layers:
            # Self-attention
            x2 = layer.self_attn(x, x, x, key_padding_mask=mask, need_weights=True)
            if isinstance(x2, tuple):
                attn_weights = x2[1]  # [B, n_heads, T, T] or None
                x2 = x2[0]
            else:
                attn_weights = None

            if attn_weights is not None:
                w = attn_weights[0].detach().cpu().numpy()
                if w.ndim == 3:
                    w = w.mean(axis=0)  # average over heads
                all_attn.append(w)

            # Rest of the encoder layer (residual, norm, FFN, norm)
            x = layer.dropout1(x2) + x
            x = layer.norm1(x)
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.dropout2(x2) + x
            x = layer.norm2(x)

        if all_attn:
            return np.stack(all_attn, axis=0)  # [n_layers, T, T]
    except Exception as e:
        warnings.warn(f"Attention extraction failed: {e}")
    return None

def export_csv(data: Dict, filename: str, output_dir: Path) -> None:
    """Export a dict of lists to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    keys = list(data.keys())
    n_rows = len(data[keys[0]])

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n_rows):
            writer.writerow([data[k][i] for k in keys])

    print(f"  ✓ CSV saved: {filepath} ({n_rows} rows)")

def export_attention_csv(stream_stats: Dict, output_dir: Path) -> None:
    """Export attention/stream statistics to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "attention_stats.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stream", "mean_l2_norm", "std_l2_norm", "relative_contribution", "n_samples"])
        for s in ["lp", "pha", "amp", "dop"]:
            ss = stream_stats.get(s, {})
            writer.writerow([s, ss.get("mean", 0), ss.get("std", 0),
                             ss.get("relative", 0), ss.get("n", 0)])

    print(f"  ✓ CSV saved: {filepath}")

def _setup_plot_style() -> None:
    sns.set_style("whitegrid")
    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

def plot_bar_importance(
    ablation_results: Optional[Dict],
    permutation_results: Optional[Dict],
    output_dir: Path,
) -> None:
    """Grouped bar chart: ablation vs permutation importance per group."""
    _setup_plot_style()

    # Aggregate permutation results to group level
    ablation_groups: Dict[str, float] = {}
    permutation_groups: Dict[str, float] = {}
    group_labels: List[str] = []

    if ablation_results is not None:
        for i, name in enumerate(ablation_results["group_name"]):
            ablation_groups[name] = ablation_results["mse_delta_mean"][i]
            if name not in group_labels:
                group_labels.append(name)

    if permutation_results is not None:
        # Aggregate per-channel permutation → group-level mean
        perm_by_group: Dict[str, List[float]] = {}
        for i, gname in enumerate(permutation_results["group_name"]):
            if gname not in perm_by_group:
                perm_by_group[gname] = []
            perm_by_group[gname].append(permutation_results["mse_delta_mean"][i])
        for gname, vals in perm_by_group.items():
            permutation_groups[gname] = float(np.mean(vals))

    # Align groups
    all_groups = group_labels if group_labels else list(ablation_groups.keys())
    abl_vals = [ablation_groups.get(g, 0.0) for g in all_groups]
    perm_vals = [permutation_groups.get(g, 0.0) for g in all_groups]

    # Shorten labels for readability
    short_labels = []
    for g in all_groups:
        if g.startswith("Doppler band"):
            short_labels.append(g.replace("Doppler band ", "Dop "))
        elif len(g) > 25:
            short_labels.append(g[:22] + "...")
        else:
            short_labels.append(g)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_groups))
    width = 0.35

    bars1 = ax.bar(x - width / 2, abl_vals, width, label="Ablation (zero-out)",
                   color="#4472C4", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, perm_vals, width, label="Permutation (shuffle)",
                   color="#ED7D31", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Δ MSE (higher → more important)")
    ax.set_title("Feature Group Importance: Ablation vs Permutation")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if abs(h) > 0.0001:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0002,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for bar in bars2:
        h = bar.get_height()
        if abs(h) > 0.0001:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0002,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    filepath = output_dir / "bar_importance.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Plot saved: {filepath}")

def plot_permutation_heatmap(
    permutation_results: Dict,
    output_dir: Path,
) -> None:
    """Heatmap: rows = channels, columns = groups, color = importance."""
    if permutation_results is None:
        return

    _setup_plot_style()

    # Build matrix: channels × groups
    group_names_ordered = []
    for gname in permutation_results["group_name"]:
        if gname not in group_names_ordered:
            group_names_ordered.append(gname)

    # Collect channel importances per group
    channel_data: List[Tuple[str, str, float]] = []
    for i, gname in enumerate(permutation_results["group_name"]):
        ch_name = permutation_results["channel_name"][i]
        mse_delta = permutation_results["mse_delta_mean"][i]
        channel_data.append((gname, ch_name, mse_delta))

    # Build matrix: max channels per group
    max_ch_per_group = {}
    for gname in group_names_ordered:
        max_ch_per_group[gname] = sum(1 for g, _, _ in channel_data if g == gname)

    max_ch = max(max_ch_per_group.values()) if max_ch_per_group else 1
    n_groups = len(group_names_ordered)

    matrix = np.full((max_ch, n_groups), np.nan)
    ch_labels = []

    for gi, gname in enumerate(group_names_ordered):
        group_channels = [(ch, val) for g, ch, val in channel_data if g == gname]
        # Sort by importance within group
        group_channels.sort(key=lambda x: x[1], reverse=True)
        for ci, (ch_name, val) in enumerate(group_channels):
            if ci < max_ch:
                matrix[ci, gi] = val
                if gi == 0:
                    ch_labels.append(ch_name[:18])

    # Truncate labels
    ch_labels = ch_labels[:max_ch]

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.2), max(6, max_ch * 0.3)))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap="RdBu_r", interpolation="nearest")

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([g[:18] for g in group_names_ordered],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(max_ch))
    ax.set_yticklabels(ch_labels if len(ch_labels) == max_ch else [""] * max_ch, fontsize=7)
    ax.set_title("Per-Channel Permutation Importance (Δ MSE)")
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Channel (sorted by importance within group)")

    plt.colorbar(im, ax=ax, fraction=0.046, label="Δ MSE")
    plt.tight_layout()
    filepath = output_dir / "permutation_heatmap.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Plot saved: {filepath}")

def plot_radar_importance(
    ablation_results: Optional[Dict],
    permutation_results: Optional[Dict],
    output_dir: Path,
) -> None:
    """Radar/spider chart comparing relative group contributions."""
    _setup_plot_style()

    # Aggregate to group level
    groups: List[str] = []
    abl_vals: List[float] = []
    perm_vals: List[float] = []

    if ablation_results is not None:
        for i, gname in enumerate(ablation_results["group_name"]):
            groups.append(gname[:20])
            abl_vals.append(ablation_results["mse_delta_mean"][i])

    if permutation_results is not None:
        perm_by_group: Dict[str, List[float]] = {}
        for i, gname in enumerate(permutation_results["group_name"]):
            if gname not in perm_by_group:
                perm_by_group[gname] = []
            perm_by_group[gname].append(permutation_results["mse_delta_mean"][i])
        for gname in groups:
            if gname in perm_by_group:
                perm_vals.append(float(np.mean(perm_by_group[gname])))
            else:
                perm_vals.append(0.0)

    if not groups:
        return

    # Normalize to [0, 1] for radar
    all_vals = [v for v in abl_vals + perm_vals if not np.isnan(v) and v > -1e9]
    if not all_vals:
        return
    vmin = min(all_vals)
    vmax = max(all_vals)
    if vmax - vmin < 1e-8:
        vmax = vmin + 0.01

    abl_norm = [(v - vmin) / (vmax - vmin) for v in abl_vals]
    perm_norm = [(v - vmin) / (vmax - vmin) for v in perm_vals]

    n = len(groups)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    ax.plot(angles, abl_norm + abl_norm[:1], "o-", linewidth=2, color="#4472C4",
            label="Ablation", markersize=4)
    ax.fill(angles, abl_norm + abl_norm[:1], alpha=0.15, color="#4472C4")

    if any(v != 0 for v in perm_norm):
        ax.plot(angles, perm_norm + perm_norm[:1], "s--", linewidth=2, color="#ED7D31",
                label="Permutation", markersize=4)
        ax.fill(angles, perm_norm + perm_norm[:1], alpha=0.1, color="#ED7D31")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_title("Feature Group Importance (normalized)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_yticklabels([])

    plt.tight_layout()
    filepath = output_dir / "radar_importance.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Plot saved: {filepath}")

def plot_scatter_variance(
    permutation_results: Dict,
    test_loader: DataLoader,
    device: str,
    train_on: str,
    output_dir: Path,
) -> None:
    """Scatter: per-channel feature variance vs. importance score."""
    if permutation_results is None:
        return

    _setup_plot_style()

    source_key = "unvoiced" if train_on == "unvoiced" else "voiced"

    # Compute per-channel variance from the data
    ch_variances: Dict[str, float] = {}
    ch_counts: Dict[str, int] = {}

    for batch in test_loader:
        for stream_key, tensor_key in [("lp", "lp"), ("pha", "pha"), ("amp", "amp"), ("dop", "dop")]:
            data = batch[source_key][tensor_key]
            for b in range(data.size(0)):
                # Get valid length
                lengths_t = batch[source_key]["lengths"]
                valid_t = int(lengths_t[b].item())
                if valid_t <= 1:
                    continue
                for c in range(data.size(2)):
                    ch_name = f"{stream_key}_{c}"
                    var = float(data[b, :valid_t, c].var().item())
                    if ch_name not in ch_variances:
                        ch_variances[ch_name] = 0.0
                        ch_counts[ch_name] = 0
                    ch_variances[ch_name] += var
                    ch_counts[ch_name] += 1

    # Average variances
    for k in ch_variances:
        if ch_counts[k] > 0:
            ch_variances[k] /= ch_counts[k]

    # Match permutation results to channel variances
    points: List[Tuple[float, float, str]] = []
    for i, gname in enumerate(permutation_results["group_name"]):
        ch_name = permutation_results["channel_name"][i]
        mse_delta = permutation_results["mse_delta_mean"][i]
        # Map channel name to variance key
        var_key = None
        for stream_key in ["lp", "pha", "amp", "dop"]:
            if ch_name.startswith(stream_key) or gname.lower().find(stream_key) >= 0:
                # Try to find matching variance
                for vk in ch_variances:
                    if vk.startswith(f"{stream_key}_"):
                        var_key = vk
                        break
                break
        if var_key is not None and var_key in ch_variances:
            points.append((ch_variances[var_key], mse_delta, gname))

    if not points:
        print("  [scatter] No matching variance data found, skipping scatter plot.")
        return

    vars_arr = np.array([p[0] for p in points])
    imps_arr = np.array([p[1] for p in points])
    groups_arr = [p[2] for p in points]

    # Color by group
    unique_groups = sorted(set(groups_arr))
    cmap = plt.get_cmap("tab10", len(unique_groups))
    group_to_color = {g: cmap(i) for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for g in unique_groups:
        mask = [gr == g for gr in groups_arr]
        ax.scatter(vars_arr[mask], imps_arr[mask], c=[group_to_color[g]],
                   label=g[:25], alpha=0.7, edgecolors="white", linewidth=0.3, s=40)

    ax.set_xlabel("Feature Variance (log scale)")
    ax.set_ylabel("Δ MSE (permutation importance)")
    ax.set_title("Feature Variance vs. Importance")
    ax.set_xscale("log")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    filepath = output_dir / "scatter_variance.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Plot saved: {filepath}")

def plot_attention_maps(attention_data: Dict, output_dir: Path) -> None:
    """Plot self-attention heatmaps for representative samples."""
    maps = attention_data.get("attention_maps", [])
    if not maps:
        print("  [attention] No attention maps to plot.")
        return

    _setup_plot_style()
    n_layers = maps[0].shape[0]
    n_samples = len(maps)

    for si, attn in enumerate(maps):
        fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3))
        if n_layers == 1:
            axes = [axes]

        for li in range(n_layers):
            ax = axes[li]
            im = ax.imshow(attn[li], aspect="auto", origin="lower",
                           cmap="viridis", interpolation="nearest")
            ax.set_title(f"L{li + 1}")
            ax.set_xlabel("Key position" if li == n_layers // 2 else "")
            ax.set_ylabel("Query position" if li == 0 else "")

        plt.suptitle(f"Self-Attention Maps — Sample {si + 1}", y=1.02)
        plt.tight_layout()
        filepath = output_dir / f"attention_sample_{si + 1}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Plot saved: {filepath}")

def main() -> None:
    args = build_parser().parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:5"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[setup] Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output directory: {output_dir}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load data
    test_loader = load_test_loader(
        args.data_dir,
        args.voiced_path,
        args.unvoiced_path,
        args.textgrid_path,
        args.n_samples,
        device,
        args.seed,
    )

    # Run methods
    ablation_results: Optional[Dict] = None
    permutation_results: Optional[Dict] = None
    attention_results: Optional[Dict] = None

    if args.method in ("ablation", "all"):
        ablation_results = run_ablation(
            model, test_loader, device, args.train_on,
            doppler_subsplits=args.doppler_subsplit,
        )
        export_csv(ablation_results, "ablation_importance.csv", output_dir)

    if args.method in ("permutation", "all"):
        permutation_results = run_permutation(
            model, test_loader, device, args.train_on,
            doppler_block_size=args.doppler_block_size,
        )
        export_csv(permutation_results, "permutation_importance.csv", output_dir)

    if args.method in ("attention", "all"):
        attention_results = run_attention_analysis(
            model, test_loader, device, args.train_on,
        )
        export_attention_csv(attention_results["stream_stats"], output_dir)
        plot_attention_maps(attention_results, output_dir)

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    plot_bar_importance(ablation_results, permutation_results, output_dir)
    plot_permutation_heatmap(permutation_results, output_dir)
    plot_radar_importance(ablation_results, permutation_results, output_dir)
    plot_scatter_variance(permutation_results, test_loader, device, args.train_on, output_dir)

    print("\n" + "=" * 60)
    print(f"All results saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()