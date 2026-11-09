import warnings
import numpy as np
import torch
from torch import Tensor
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# WER / CER via ASR
# ---------------------------------------------------------------------------

def compute_wer_cer(
    pred_audio: np.ndarray,
    gt_text: str,
    sample_rate: int = 44100,
    asr_model_name: str = "openai/whisper-base",
    device: str = "cpu",
    asr_model=None,
) -> dict:
    """Word Error Rate (WER) and Character Error Rate (CER) via ASR."""
    try:
        import whisper
    except ImportError:
        warnings.warn("whisper not installed; WER/CER unavailable.")
        return {"wer": float("nan"), "cer": float("nan"), "hypothesis": ""}

    if asr_model is None:
        asr_model = whisper.load_model(
            asr_model_name.replace("openai/", "").replace("-", "."),
            device=device,
        )
    # Whisper expects float32 audio at 16 kHz
    target_sr = 16000
    if sample_rate != target_sr:
        from metrics.perceptual import _resample
        pred_audio = _resample(pred_audio, sample_rate, target_sr)

    result = asr_model.transcribe(pred_audio.astype(np.float32), language="en")
    hypothesis = result["text"].strip()

    # WER / CER calculation
    wer = _word_error_rate(gt_text, hypothesis)
    cer = _char_error_rate(gt_text, hypothesis)

    return {"wer": wer, "cer": cer, "hypothesis": hypothesis}

def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard word error rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    return _levenshtein_error_rate(ref_words, hyp_words)

def _char_error_rate(reference: str, hypothesis: str) -> float:
    """Standard character error rate."""
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))
    return _levenshtein_error_rate(ref_chars, hyp_chars)

def _levenshtein_error_rate(ref: list, hyp: list) -> float:
    """Edit-distance-based error rate."""
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return 1.0

    # DP table
    d = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + cost,
            )

    return float(d[n, m]) / float(n)

def compute_per(
    pred_phonemes: Tensor,
    gt_phonemes: Tensor,
    pred_lengths: Optional[Tensor] = None,
    gt_lengths: Optional[Tensor] = None,
    phoneme_inventory: Optional[List[str]] = None,
) -> float:
    """Phoneme Error Rate (PER) — frame-level classification accuracy."""
    if pred_phonemes.dim() == 3:
        pred_indices = pred_phonemes.argmax(dim=-1)
    else:
        pred_indices = pred_phonemes

    total_errors = 0
    total_frames = 0

    B = pred_indices.size(0)
    for b in range(B):
        p_len = int(pred_lengths[b].item()) if pred_lengths is not None else pred_indices.size(1)
        g_len = int(gt_lengths[b].item()) if gt_lengths is not None else gt_phonemes.size(1)
        valid_len = min(p_len, g_len)
        if valid_len == 0:
            continue

        pred_valid = pred_indices[b, :valid_len]
        gt_valid = gt_phonemes[b, :valid_len]

        errors = (pred_valid != gt_valid).sum().item()
        total_errors += errors
        total_frames += valid_len

    if total_frames == 0:
        return 1.0
    return float(total_errors) / float(total_frames)

# ---------------------------------------------------------------------------
# Phoneme Confusion Matrix
# ---------------------------------------------------------------------------
def compute_phoneme_confusion_matrix(
    pred_phonemes: Tensor,
    gt_phonemes: Tensor,
    pred_lengths: Optional[Tensor] = None,
    gt_lengths: Optional[Tensor] = None,
    num_classes: int = 97,
    phoneme_inventory: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Phoneme confusion matrix."""
    if pred_phonemes.dim() == 3:
        pred_indices = pred_phonemes.argmax(dim=-1)
    else:
        pred_indices = pred_phonemes

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    B = pred_indices.size(0)
    for b in range(B):
        p_len = int(pred_lengths[b].item()) if pred_lengths is not None else pred_indices.size(1)
        g_len = int(gt_lengths[b].item()) if gt_lengths is not None else gt_phonemes.size(1)
        valid_len = min(p_len, g_len)
        if valid_len == 0:
            continue

        for t in range(valid_len):
            gt_idx = int(gt_phonemes[b, t].item())
            pred_idx = int(pred_indices[b, t].item())
            if 0 <= gt_idx < num_classes and 0 <= pred_idx < num_classes:
                cm[gt_idx, pred_idx] += 1

    labels = phoneme_inventory if phoneme_inventory else [str(i) for i in range(num_classes)]
    return cm, labels

def compute_all_intelligibility(
    pred_phonemes: Tensor,
    gt_phonemes: Tensor,
    pred_lengths: Optional[Tensor],
    gt_lengths: Optional[Tensor],
    phoneme_inventory: Optional[List[str]] = None,
) -> dict:
    """Compute all Group 3 intelligibility metrics."""
    per = compute_per(
        pred_phonemes, gt_phonemes, pred_lengths, gt_lengths, phoneme_inventory
    )
    cm, labels = compute_phoneme_confusion_matrix(
        pred_phonemes, gt_phonemes, pred_lengths, gt_lengths,
        num_classes=len(phoneme_inventory) if phoneme_inventory else pred_phonemes.size(-1),
        phoneme_inventory=phoneme_inventory,
    )
    return {
        "per": per,
        "confusion_matrix": cm,
        "confusion_labels": labels,
    }