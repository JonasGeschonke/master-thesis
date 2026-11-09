import torch
import warnings
import sys
import numpy as np

from pathlib import Path
from torch import Tensor
from typing import List, Optional, Dict

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from metrics.spectral import compute_mcd, compute_mel_rmse
from metrics.perceptual import compute_stoi

def compute_dtw_alignment_cost(
    pred_mel: Tensor,
    gt_mel: Tensor,
    normalize: bool = True,
) -> float:
    """Normalized cumulative DTW distance between predicted and GT mel sequences."""
    pred_np = pred_mel.detach().cpu().numpy()
    gt_np = gt_mel.detach().cpu().numpy()

    # Pairwise Euclidean distances between all frames
    # distances[i, j] = ||gt_mel[i] - pred_mel[j]||_2
    T_gt = gt_np.shape[0]
    T_pred = pred_np.shape[0]
    distances = np.zeros((T_gt, T_pred))
    for i in range(T_gt):
        diff = gt_np[i:i + 1, :] - pred_np
        distances[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))

    # Compute DTW
    dtw_matrix = _time_warp(distances)
    cost = dtw_matrix[-1, -1]

    if normalize:
        # Backtrack to count path length
        path_len = _count_path(dtw_matrix)
        cost = cost / max(path_len, 1)

    return float(cost)

def _time_warp(costs: np.ndarray) -> np.ndarray:
    """Classic DTW accumulation matrix."""
    T_gt, T_pred = costs.shape
    dtw = np.full((T_gt, T_pred), np.inf)
    dtw[0, 0] = costs[0, 0]

    for i in range(1, T_gt):
        dtw[i, 0] = dtw[i - 1, 0] + costs[i, 0]
    for j in range(1, T_pred):
        dtw[0, j] = dtw[0, j - 1] + costs[0, j]

    for i in range(1, T_gt):
        for j in range(1, T_pred):
            dtw[i, j] = costs[i, j] + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )
    return dtw

def _count_path(dtw: np.ndarray) -> int:
    """Count the number of steps in the optimal DTW path."""
    i, j = dtw.shape[0] - 1, dtw.shape[1] - 1
    steps = 0
    while i > 0 or j > 0:
        steps += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            moves = [
                (i - 1, j, dtw[i - 1, j]),
                (i, j - 1, dtw[i, j - 1]),
                (i - 1, j - 1, dtw[i - 1, j - 1]),
            ]
            i, j, _ = min(moves, key=lambda x: x[2])
    return steps + 1

# ---------------------------------------------------------------------------
# Per-Phoneme MCD / STOI
# ---------------------------------------------------------------------------
def compute_per_phoneme_metrics(
    pred_mel: Tensor,
    gt_mel: Tensor,
    phoneme_labels: Tensor,  # [T] frame-level phoneme indices
    phoneme_classes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute MCD and mel RMSE per phoneme class."""

    pred_np = pred_mel.detach().cpu().numpy()
    gt_np = gt_mel.detach().cpu().numpy()
    labels_np = phoneme_labels.detach().cpu().numpy()

    unique_labels = np.unique(labels_np)
    results = {}

    for label in unique_labels:
        mask = labels_np == label
        n_frames = int(mask.sum())
        if n_frames < 3:
            continue

        pred_sub = torch.from_numpy(pred_np[mask])
        gt_sub = torch.from_numpy(gt_np[mask])

        label_name = (
            phoneme_classes[int(label)]
            if phoneme_classes and int(label) < len(phoneme_classes)
            else str(int(label))
        )

        try:
            mcd_val = compute_mcd(pred_sub, gt_sub)
        except Exception:
            mcd_val = float("nan")

        rmse_val = compute_mel_rmse(pred_sub, gt_sub, per_band=False)

        results[label_name] = {
            "mcd": mcd_val,
            "rmse": float(rmse_val),
            "n_frames": n_frames,
        }

    return results

def compute_per_phoneme_stoi(
    pred_audio: np.ndarray,
    gt_audio: np.ndarray,
    phoneme_labels: Tensor,
    sample_rate: int = 44100,
    hop_length: int = 256,
    phoneme_classes: Optional[List[str]] = None,
    min_frames: int = 3,
) -> Dict[str, float]:
    """STOI computed per phoneme class on the corresponding audio segments."""

    labels_np = phoneme_labels.detach().cpu().numpy()
    results: Dict[str, float] = {}

    for label in np.unique(labels_np):
        frame_idx = np.where(labels_np == label)[0]
        n_frames = int(len(frame_idx))
        if n_frames < min_frames:
            continue

        start = int(frame_idx[0]) * hop_length
        end = int(frame_idx[-1] + 1) * hop_length
        pred_seg = pred_audio[start:end]
        gt_seg = gt_audio[start:end]
        if len(pred_seg) < int(sample_rate * 0.4) or len(gt_seg) < int(sample_rate * 0.4):
            continue

        label_name = (
            phoneme_classes[int(label)]
            if phoneme_classes and int(label) < len(phoneme_classes)
            else str(int(label))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                results[label_name] = compute_stoi(pred_seg, gt_seg, sample_rate)
            except Exception:
                results[label_name] = float("nan")

    return results

# ---------------------------------------------------------------------------
# F0 (Pitch) Correlation
# ---------------------------------------------------------------------------
def compute_f0_correlation(
    pred_audio: np.ndarray,
    gt_audio: np.ndarray,
    sample_rate: int = 44100,
    f0_min: float = 50.0,
    f0_max: float = 400.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> float:
    """Pearson r between predicted and GT fundamental frequency (F0) contours."""
    f0_pred = _extract_f0_contour(pred_audio, sample_rate, f0_min, f0_max, frame_ms, hop_ms)
    f0_gt = _extract_f0_contour(gt_audio, sample_rate, f0_min, f0_max, frame_ms, hop_ms)

    # Align lengths
    min_len = min(len(f0_pred), len(f0_gt))
    f0_pred = f0_pred[:min_len]
    f0_gt = f0_gt[:min_len]

    # Only use frames where BOTH have valid F0 (voiced)
    voiced_mask = (f0_pred > 0) & (f0_gt > 0)
    if voiced_mask.sum() < 10:
        return float("nan")

    f0_pred_v = f0_pred[voiced_mask]
    f0_gt_v = f0_gt[voiced_mask]

    return float(np.corrcoef(f0_pred_v, f0_gt_v)[0, 1])

def _extract_f0_contour(
    audio: np.ndarray,
    sr: int,
    f0_min: float,
    f0_max: float,
    frame_ms: float,
    hop_ms: float,
) -> np.ndarray:
    """Simple autocorrelation-based F0 extraction."""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    f0_contour = np.zeros(n_frames)

    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)

    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        if len(frame) < max_lag:
            continue

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        if len(autocorr) <= max_lag:
            continue

        # Search for peak in valid F0 range
        search_range = autocorr[min_lag:min(max_lag, len(autocorr))]
        if len(search_range) == 0:
            continue

        peak_idx = np.argmax(search_range)
        peak_val = search_range[peak_idx]

        # Voicing threshold: peak must be significant relative to zero-lag
        if autocorr[0] > 0 and peak_val / autocorr[0] > 0.3:
            f0_contour[i] = sr / (min_lag + peak_idx)

    return f0_contour


# ---------------------------------------------------------------------------
# Voiced/Unvoiced Detection Accuracy
# ---------------------------------------------------------------------------
def compute_voiced_unvoiced_accuracy(
    pred_f0: Optional[np.ndarray] = None,
    gt_f0: Optional[np.ndarray] = None,
    pred_audio: Optional[np.ndarray] = None,
    gt_audio: Optional[np.ndarray] = None,
    sample_rate: int = 44100,
) -> float:
    """Binary classification accuracy of voiced vs. unvoiced frames."""
    if pred_f0 is None and pred_audio is not None:
        pred_f0 = _extract_f0_contour(pred_audio, sample_rate, 50, 400, 25, 10)
    if gt_f0 is None and gt_audio is not None:
        gt_f0 = _extract_f0_contour(gt_audio, sample_rate, 50, 400, 25, 10)

    if pred_f0 is None or gt_f0 is None:
        return float("nan")

    min_len = min(len(pred_f0), len(gt_f0))
    pred_voiced = (pred_f0[:min_len] > 0).astype(int)
    gt_voiced = (gt_f0[:min_len] > 0).astype(int)

    return float(np.mean(pred_voiced == gt_voiced))

# ---------------------------------------------------------------------------
# Phoneme Duration Correlation
# ---------------------------------------------------------------------------
def compute_phoneme_duration_correlation(
    pred_phoneme_labels: Tensor,
    gt_phoneme_labels: Tensor,
) -> float:
    """Correlation between predicted and GT phoneme segment durations."""
    pred_durations = _get_segment_durations(pred_phoneme_labels)
    gt_durations = _get_segment_durations(gt_phoneme_labels)

    if len(pred_durations) < 3 or len(gt_durations) < 3:
        return float("nan")

    # Align by segment count (take first N segments)
    n = min(len(pred_durations), len(gt_durations))
    pred_d = np.array(pred_durations[:n])
    gt_d = np.array(gt_durations[:n])

    if np.std(pred_d) < 1e-6 or np.std(gt_d) < 1e-6:
        return float("nan")

    return float(np.corrcoef(pred_d, gt_d)[0, 1])

def _get_segment_durations(labels: Tensor) -> List[int]:
    """Extract segment durations from a sequence of phoneme labels."""
    labels_np = labels.detach().cpu().numpy()
    durations = []
    if len(labels_np) == 0:
        return durations

    current_label = labels_np[0]
    count = 1
    for i in range(1, len(labels_np)):
        if labels_np[i] == current_label:
            count += 1
        else:
            durations.append(count)
            current_label = labels_np[i]
            count = 1
    durations.append(count)
    return durations

def compute_all_ssi_specific(
    pred_mel: Tensor,
    gt_mel: Tensor,
    pred_phonemes: Tensor,
    gt_phonemes: Tensor,
    pred_audio: np.ndarray,
    gt_audio: np.ndarray,
    sample_rate: int = 44100,
    phoneme_inventory: Optional[List[str]] = None,
) -> dict:
    """Compute all Group 4 SSI-specific metrics."""
    dtw_cost = compute_dtw_alignment_cost(pred_mel, gt_mel)

    # Per-phoneme metrics need frame-level phoneme labels
    per_phoneme = {}
    if gt_phonemes is not None:
        per_phoneme = compute_per_phoneme_metrics(
            pred_mel, gt_mel, gt_phonemes, phoneme_inventory
        )

    f0_corr = compute_f0_correlation(pred_audio, gt_audio, sample_rate)
    vu_acc = compute_voiced_unvoiced_accuracy(
        pred_audio=pred_audio, gt_audio=gt_audio, sample_rate=sample_rate
    )
    dur_corr = compute_phoneme_duration_correlation(
        pred_phonemes.argmax(dim=-1) if pred_phonemes.dim() == 3 else pred_phonemes,
        gt_phonemes,
    )

    return {
        "dtw_cost": dtw_cost,
        "per_phoneme_metrics": per_phoneme,
        "f0_corr": f0_corr,
        "voiced_unvoiced_acc": vu_acc,
        "duration_corr": dur_corr,
    }