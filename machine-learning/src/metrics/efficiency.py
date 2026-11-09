import os
import time
import warnings
import torch
import numpy as np
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Optional, Tuple

def count_parameters(model: Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compute_flops(
    model: Module,
    input_shape_lp: Tuple[int, int, int] = (1, 200, 34),
    input_shape_pha: Tuple[int, int, int] = (1, 200, 1),
    input_shape_amp: Tuple[int, int, int] = (1, 200, 1),
    input_shape_dop: Tuple[int, int, int] = (1, 200, 1024),
    device: str = "cpu",
) -> Optional[int]:
    """Estimate FLOPs for a single forward pass."""
    try:
        from fvcore.nn import FlopCountAnalysis

        model = model.to(device)
        model.eval()

        lp = torch.randn(*input_shape_lp, device=device)
        pha = torch.randn(*input_shape_pha, device=device)
        amp = torch.randn(*input_shape_amp, device=device)
        dop = torch.randn(*input_shape_dop, device=device)
        lengths = torch.full((input_shape_lp[0],), input_shape_lp[1], dtype=torch.long, device=device)

        with torch.no_grad():
            flops = FlopCountAnalysis(model, (lp, pha, amp, dop, lengths))
            return flops.total()
    except ImportError:
        warnings.warn("fvcore not installed; FLOPs unavailable. Install: pip install fvcore")
        return None

def compute_rtf(
    model: Module,
    dataloader: DataLoader,
    hyperparameters,
    n_batches: int = 10,
    device: str = "cpu",
    train_on: str = "voiced",
) -> float:
    """Real-Time Factor — processing time divided by audio duration."""
    model = model.to(device)
    model.eval()

    total_processing_time = 0.0
    total_audio_duration = 0.0
    hop_length = getattr(hyperparameters, 'hop_length', 256)
    sample_rate = getattr(hyperparameters, 'sample_rate', 44100)
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= n_batches:
                break

            src_key = train_on if train_on in ('voiced', 'unvoiced') else 'voiced'
            lp_seq = batch[src_key]['lp'].to(device)
            pha_seq = batch[src_key]['pha'].to(device)
            amp_seq = batch[src_key]['amp'].to(device)
            dop_seq = batch[src_key]['dop'].to(device)
            lengths = batch[src_key]['lengths'].to(device)

            # Time the forward pass
            if isinstance(device, str) and device.startswith("cuda"):
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            _, output_lengths, _ = model(lp_seq, pha_seq, amp_seq, dop_seq, lengths=lengths)

            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            processing_time = t_end - t_start

            # Audio duration: output frames × hop_length / sample_rate
            audio_duration = output_lengths.sum().item() * hop_length / sample_rate

            total_processing_time += processing_time
            total_audio_duration += audio_duration
            count += 1

    if count == 0 or total_audio_duration == 0:
        return float("nan")

    return total_processing_time / total_audio_duration

def measure_peak_memory(
    model: Module,
    dataloader: DataLoader,
    hyperparameters,
    n_batches: int = 5,
    device: str = "cuda",
    train_on: str = "voiced",
) -> float:
    """Peak GPU memory allocated during a forward pass, in MB."""
    if not isinstance(device, str) or not device.startswith("cuda") or not torch.cuda.is_available():
        warnings.warn("Peak memory measurement requires CUDA.")
        return float("nan")

    model = model.to(device)
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            src_key = train_on if train_on in ('voiced', 'unvoiced') else 'voiced'
            lp_seq = batch[src_key]['lp'].to(device)
            pha_seq = batch[src_key]['pha'].to(device)
            amp_seq = batch[src_key]['amp'].to(device)
            dop_seq = batch[src_key]['dop'].to(device)
            lengths = batch[src_key]['lengths'].to(device)
            _ = model(lp_seq, pha_seq, amp_seq, dop_seq, lengths=lengths)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 * 1024)  # Convert bytes to MB

def measure_model_size(checkpoint_path: str) -> float:
    """Model size on disk in MB."""
    try:
        return os.path.getsize(checkpoint_path) / (1024 * 1024)
    except OSError:
        return float("nan")

def measure_throughput(
    model: Module,
    dataloader: DataLoader,
    hyperparameters,
    n_batches: int = 20,
    device: str = "cpu",
    train_on: str = "voiced",
) -> float:
    """Inference throughput in samples per second (batch_size=1)."""
    model = model.to(device)
    model.eval()

    total_samples = 0
    total_time = 0.0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            src_key = train_on if train_on in ('voiced', 'unvoiced') else 'voiced'
            lp_seq = batch[src_key]['lp'].to(device)
            pha_seq = batch[src_key]['pha'].to(device)
            amp_seq = batch[src_key]['amp'].to(device)
            dop_seq = batch[src_key]['dop'].to(device)
            lengths = batch[src_key]['lengths'].to(device)

            batch_size = lp_seq.size(0)

            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            _ = model(lp_seq, pha_seq, amp_seq, dop_seq, lengths=lengths)

            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            total_time += (t_end - t_start)
            total_samples += batch_size

    if total_time == 0:
        return float("nan")
    return total_samples / total_time

def compute_all_efficiency(
    model: Module,
    dataloader: DataLoader,
    hyperparameters,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    train_on: str = "voiced",
) -> dict:
    """Compute all Group 5 efficiency metrics."""
    total_p, trainable_p = count_parameters(model)

    return {
        "total_params": total_p,
        "trainable_params": trainable_p,
        "flops": compute_flops(model, device=device),
        "rtf": compute_rtf(model, dataloader, hyperparameters, device=device, train_on=train_on),
        "peak_memory_mb": measure_peak_memory(model, dataloader, hyperparameters, device=device, train_on=train_on),
        "model_size_mb": measure_model_size(checkpoint_path) if checkpoint_path else float("nan"),
        "throughput_sps": measure_throughput(model, dataloader, hyperparameters, device=device, train_on=train_on),
    }