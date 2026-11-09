"""
SSI Reconstruction Model Comparison Metrics
============================================
Comprehensive multi-group metric framework for comparing silent speech
reconstruction models along fidelity, perceptual quality, intelligibility,
SSI-specific mapping quality, computational efficiency, and training dynamics.

Based on: specs/metrics.md
"""

from metrics.spectral import (
    compute_mcd,
    compute_lsd,
    compute_mel_rmse,
    compute_pearson_per_band,
)
from metrics.perceptual import (
    compute_stoi,
    compute_pesq,
    compute_dnsmos,
)
from metrics.intelligibility import (
    compute_wer_cer,
    compute_per,
    compute_phoneme_confusion_matrix,
)
from metrics.ssi_specific import (
    compute_dtw_alignment_cost,
    compute_per_phoneme_metrics,
    compute_f0_correlation,
    compute_voiced_unvoiced_accuracy,
    compute_phoneme_duration_correlation,
)
from metrics.efficiency import (
    count_parameters,
    compute_flops,
    compute_rtf,
    measure_peak_memory,
    measure_model_size,
    measure_throughput,
)

__all__ = [
    # Spectral
    "compute_mcd", "compute_lsd", "compute_mel_rmse", "compute_pearson_per_band",
    # Perceptual
    "compute_stoi", "compute_pesq", "compute_dnsmos",
    # Intelligibility
    "compute_wer_cer", "compute_per", "compute_phoneme_confusion_matrix",
    # SSI-specific
    "compute_dtw_alignment_cost", "compute_per_phoneme_metrics",
    "compute_f0_correlation", "compute_voiced_unvoiced_accuracy",
    "compute_phoneme_duration_correlation",
    # Efficiency
    "count_parameters", "compute_flops", "compute_rtf",
    "measure_peak_memory", "measure_model_size", "measure_throughput",
]
