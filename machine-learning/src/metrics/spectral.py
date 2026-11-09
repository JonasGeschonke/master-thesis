import torch
import numpy as np
from torch import Tensor

def _to_numpy_float64(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    return arr

def compute_mcd(
    pred_mel: Tensor,
    gt_mel: Tensor,
    n_mfcc: int = 13,
    eps: float = 1e-10,
) -> float:
    """Mel Cepstral Distortion (MCD) in dB."""
    pred_np = _to_numpy_float64(pred_mel)
    gt_np = _to_numpy_float64(gt_mel)

    if pred_np.size == 0 or gt_np.size == 0 or pred_np.shape != gt_np.shape:
        if pred_np.size and gt_np.size:
            min_t = min(pred_np.shape[0], gt_np.shape[0])
            pred_np = pred_np[:min_t]
            gt_np = gt_np[:min_t]
        else:
            return float("nan")

    if pred_np.ndim != 2 or gt_np.ndim != 2 or pred_np.shape[1] == 0:
        return float("nan")

    # Convert log-mel to magnitude (approximate inversion)
    pred_mag = np.exp(pred_np) - 1.0
    gt_mag = np.exp(gt_np) - 1.0
    pred_mag = np.maximum(pred_mag, 0.0)
    gt_mag = np.maximum(gt_mag, 0.0)

    if not np.all(np.isfinite(pred_mag)) or not np.all(np.isfinite(gt_mag)):
        return float("nan")

    mfcc_pred = _dct_ii(pred_mag, n_mfcc)
    mfcc_gt = _dct_ii(gt_mag, n_mfcc)

    # Euclidean distance per frame, then average
    diff = mfcc_pred - mfcc_gt
    mcd_per_frame = np.sqrt(np.sum(diff ** 2, axis=1))
    if mcd_per_frame.size == 0 or not np.all(np.isfinite(mcd_per_frame)):
        return float("nan")
    mcd_db = (10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(mcd_per_frame)
    return float(mcd_db)

def _dct_ii(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    """Type-II DCT along the frequency axis (last dim)."""
    T, F = x.shape
    n_coeffs = min(n_coeffs, F)
    mfcc = np.zeros((T, n_coeffs))
    for k in range(n_coeffs):
        mfcc[:, k] = np.sum(x * np.cos(np.pi * k * (np.arange(F) + 0.5) / F), axis=1)
    return mfcc

def compute_lsd(
    pred_mel: Tensor,
    gt_mel: Tensor,
) -> float:
    """Log-Spectral Distance (LSD) in dB."""
    pred_np = _to_numpy_float64(pred_mel)
    gt_np = _to_numpy_float64(gt_mel)

    if pred_np.size == 0 or gt_np.size == 0:
        return float("nan")
    if pred_np.shape != gt_np.shape:
        min_t = min(pred_np.shape[0], gt_np.shape[0])
        pred_np = pred_np[:min_t]
        gt_np = gt_np[:min_t]
    if pred_np.ndim != 2 or gt_np.ndim != 2 or pred_np.shape[1] == 0:
        return float("nan")

    diff = pred_np - gt_np
    if not np.all(np.isfinite(diff)):
        return float("nan")
    lsd_per_frame = np.sqrt(np.mean(diff ** 2, axis=1))
    if lsd_per_frame.size == 0 or not np.all(np.isfinite(lsd_per_frame)):
        return float("nan")
    lsd_db = np.mean(lsd_per_frame)
    return float(lsd_db)

def compute_mel_rmse(
    pred_mel: Tensor,
    gt_mel: Tensor,
    per_band: bool = False,
) -> float | np.ndarray:
    """Frame-wise RMSE between predicted and ground-truth mel spectrograms."""
    pred_np = _to_numpy_float64(pred_mel)
    gt_np = _to_numpy_float64(gt_mel)

    if pred_np.size == 0 or gt_np.size == 0:
        return float("nan") if not per_band else np.full((0,), np.nan, dtype=np.float64)
    if pred_np.shape != gt_np.shape:
        min_t = min(pred_np.shape[0], gt_np.shape[0])
        pred_np = pred_np[:min_t]
        gt_np = gt_np[:min_t]
    if pred_np.ndim != 2 or gt_np.ndim != 2 or pred_np.shape[1] == 0:
        return float("nan") if not per_band else np.full((0,), np.nan, dtype=np.float64)

    sq_err = (pred_np - gt_np) ** 2
    if not np.all(np.isfinite(sq_err)):
        return float("nan") if not per_band else np.full((pred_np.shape[1],), np.nan, dtype=np.float64)
    if per_band:
        return np.sqrt(np.mean(sq_err, axis=0))
    return float(np.sqrt(np.mean(sq_err)))

def compute_pearson_per_band(
    pred_mel: Tensor,
    gt_mel: Tensor,
) -> np.ndarray:
    """Per-band Pearson correlation between predicted and GT mel energies."""
    pred_np = _to_numpy_float64(pred_mel)
    gt_np = _to_numpy_float64(gt_mel)

    if pred_np.size == 0 or gt_np.size == 0 or pred_np.shape != gt_np.shape:
        return np.array([], dtype=np.float64)
    if pred_np.ndim != 2 or gt_np.ndim != 2 or pred_np.shape[1] == 0:
        return np.array([], dtype=np.float64)

    n_mels = pred_np.shape[1]
    correlations = np.full(n_mels, np.nan, dtype=np.float64)

    for m in range(n_mels):
        p = pred_np[:, m]
        g = gt_np[:, m]
        if p.size == 0 or g.size == 0 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(g)):
            continue
        p_centered = p - np.mean(p)
        g_centered = g - np.mean(g)
        numerator = np.sum(p_centered * g_centered)
        denominator = np.sqrt(np.sum(p_centered ** 2) * np.sum(g_centered ** 2))
        if denominator > 1e-10:
            correlations[m] = numerator / denominator

    return correlations

def compute_all_spectral(
    pred_mel: Tensor,
    gt_mel: Tensor,
    n_mfcc: int = 13,
) -> dict:
    """Compute all Group 1 spectral fidelity metrics."""
    pearson = compute_pearson_per_band(pred_mel, gt_mel)
    return {
        "mcd": compute_mcd(pred_mel, gt_mel, n_mfcc=n_mfcc),
        "lsd": compute_lsd(pred_mel, gt_mel),
        "mel_rmse": compute_mel_rmse(pred_mel, gt_mel, per_band=False),
        "pearson_mean": float(np.mean(pearson)),
        "pearson_std": float(np.std(pearson)),
    }