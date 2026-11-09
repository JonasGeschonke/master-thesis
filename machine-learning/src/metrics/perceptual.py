import warnings
import numpy as np

from pystoi import stoi
from pesq import pesq
from speechmos.dnsmos import run as dnsmos_run

def compute_stoi(
    pred_audio: np.ndarray,
    gt_audio: np.ndarray,
    sample_rate: int = 44100,
) -> float:
    """Short-Time Objective Intelligibility (STOI)."""
    try:
        # pystoi expects float64
        score = stoi(
            gt_audio.astype(np.float64),
            pred_audio.astype(np.float64),
            sample_rate,
            extended=False,
        )
        return float(score)
    except ImportError:
        warnings.warn("pystoi not installed; using simplified STOI fallback.")
        return _simplified_stoi(pred_audio, gt_audio, sample_rate)

def _simplified_stoi(
    pred: np.ndarray,
    gt: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> float:
    """Simplified STOI: correlation of short-time envelope in 1/3-octave bands."""
    # Match lengths
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]

    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = max(1, (min_len - frame_len) // hop_len + 1)

    correlations = []
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        p_frame = pred[start:end]
        g_frame = gt[start:end]

        # Simple envelope via RMS
        p_env = np.sqrt(np.mean(p_frame ** 2) + 1e-10)
        g_env = np.sqrt(np.mean(g_frame ** 2) + 1e-10)
        if p_env > 1e-10 and g_env > 1e-10:
            corr = np.corrcoef(p_frame, g_frame)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    if not correlations:
        return 0.0
    return float(np.clip(np.mean(correlations), 0.0, 1.0))

def compute_pesq(
    pred_audio: np.ndarray,
    gt_audio: np.ndarray,
    sample_rate: int = 44100,
    mode: str = "wb",
) -> float:
    """Perceptual Evaluation of Speech Quality (PESQ)."""
    try:
        # Resample to 16 kHz for WB-PESQ or 8 kHz for NB-PESQ
        target_sr = 16000 if mode == "wb" else 8000
        if sample_rate != target_sr:
            pred_audio = _resample(pred_audio, sample_rate, target_sr)
            gt_audio = _resample(gt_audio, sample_rate, target_sr)
        score = pesq(target_sr, gt_audio.astype(np.float64), pred_audio.astype(np.float64), mode)
        return float(score)
    except ImportError:
        warnings.warn("pesq not installed; returning NaN for PESQ.")
        return float("nan")

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resampling."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    new_len = int(duration * target_sr)
    old_indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(old_indices, np.arange(len(audio)), audio)

def compute_dnsmos(
    pred_audio: np.ndarray,
    sample_rate: int = 44100,
) -> float:
    """Deep Noise Suppression MOS (DNSMOS) via speechmos."""
    target_sr = 16000
    if sample_rate != target_sr:
        pred_audio = _resample(pred_audio, sample_rate, target_sr)

    try:
        result = dnsmos_run(
            pred_audio,
            target_sr,
            model_type="dnsmos",
            return_df=False,
            verbose=False,
        )
        if isinstance(result, dict):
            return float(result.get("ovrl_mos", result.get("OVRL", np.nan)))
        return float(result)
    except ImportError:
        warnings.warn(
            "speechmos package not installed; DNSMOS unavailable. "
            "Install via: pip install speechmos"
        )
        return float("nan")
    except Exception as exc:
        warnings.warn(f"Failed to compute DNSMOS via speechmos: {exc}")
        return float("nan")

def compute_waveform_mse(
    pred_audio: np.ndarray,
    gt_audio: np.ndarray,
) -> float:
    """Time-domain mean squared error after length alignment."""
    min_len = min(len(pred_audio), len(gt_audio))
    if min_len == 0:
        return float("nan")
    pred = pred_audio[:min_len].astype(np.float64)
    gt = gt_audio[:min_len].astype(np.float64)
    return float(np.mean((pred - gt) ** 2))

def compute_all_perceptual(
    pred_audio: np.ndarray,
    gt_audio: np.ndarray,
    sample_rate: int = 44100,
) -> dict:
    """Compute all Group 2 perceptual quality metrics."""
    return {
        "stoi": compute_stoi(pred_audio, gt_audio, sample_rate),
        "pesq": compute_pesq(pred_audio, gt_audio, sample_rate),
        "dnsmos": compute_dnsmos(pred_audio, sample_rate),
        "waveform_mse": compute_waveform_mse(pred_audio, gt_audio),
    }