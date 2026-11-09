import os
import concurrent.futures
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor, cdist, tensor
from torch.nn.functional import log_softmax, cross_entropy, l1_loss
from numba import jit

@jit(nopython=True, nogil=True)
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw

def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2 using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        plt.matshow(visual)
        plt.show()

    return results

def decollate_tensor(tensor, lengths):
    b, s, d = tensor.size()
    tensor = tensor.view(b*s, d)
    results = []
    idx = 0
    for length in lengths:
        assert idx + length <= b * s
        results.append(tensor[idx:idx+length])
        idx += length
    return results

def pad_or_trim_sequence(sequence: Tensor, target_length: int, pad_value: int = 0) -> Tensor:
    if sequence.size(0) == target_length:
        return sequence
    if sequence.size(0) < target_length:
        pad_amount = target_length - sequence.size(0)
        pad_tensor = sequence.new_full((pad_amount,), pad_value)
        return torch.cat([sequence, pad_tensor], dim=0)
    return sequence[:target_length]

def _append_deltas(x: Tensor) -> Tensor:
    d1 = torch.diff(x, dim=0, prepend=x[:1])
    d2 = torch.diff(d1, dim=0, prepend=d1[:1])
    return torch.cat([x, d1, d2], dim=-1)


def _l1_cost_matrix(a: Tensor, b: Tensor, chunk_size: int = 256) -> Tensor:
    """Memory-efficient pairwise L1 distance matrix."""
    T_a = a.size(0)
    T_b = b.size(0)
    costs = torch.empty(T_a, T_b, device=a.device, dtype=a.dtype)
    for start in range(0, T_a, chunk_size):
        end = min(start + chunk_size, T_a)
        diff = a[start:end, None, :] - b[None, :, :]
        costs[start:end] = diff.abs().sum(dim=-1)
    return costs

# ── Parallel DTW alignment ──────────────────────────────────────────────
# `time_warp` is @jit(nogil=True): the O(T²) DP releases the GIL during
# execution, so per-sample alignments run concurrently on CPU threads rather
# than serially inside the training loop (the GPU cost matrices stay on GPU).
_ALIGN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(2, min(32, os.cpu_count() or 4))
)

def _align_from_costs(costs: Tensor) -> np.ndarray:
    """Run the DTW alignment for a single [T_gt, T_pred] cost matrix on CPU."""
    return np.array(align_from_distances(costs.T.cpu().detach().numpy()))

def dtw_loss_with_phonemes_and_alignment(
    predictions, ground_truth, pred_lengths, gt_lengths,
    phoneme_prediction, phoneme_target, phoneme_prediction_lengths, phoneme_target_lengths,
    normalize_by_sample: bool = True,
    # ── Sharpness improvements ──────────────────────────────────────────
    use_delta_cost: bool = True,
    use_band_weight: bool = True,
    use_l1_cost: bool = True,
    use_contrast_loss: bool = True,
    contrast_weight: float = 0.05,
    use_post_alignment_l1: bool = True,
    post_alignment_weight: float = 0.3,
    band_decay: float = 1.5,
) -> tuple[Tensor, Tensor, float, list]:
    device = predictions.device
    batch_size = predictions.size(0)
    n_mels = predictions.size(-1)

    # ── (2) Pre-compute band weighting ─────────────────────────────────
    if use_band_weight:
        # Exponential decay: lowest mel band weight=~1.0, highest ~0.3
        band_w = torch.exp(
            -torch.arange(n_mels, device=device).float() / (n_mels / band_decay)
        )
    else:
        band_w = torch.ones(n_mels, device=device)

    predictions = [predictions[i][:pred_lengths[i]] for i in range(batch_size)]
    ground_truth = [ground_truth[i][:gt_lengths[i]] for i in range(batch_size)]
    pred_phone = log_softmax(phoneme_prediction, -1)
    pred_phone = [pred_phone[i][:pred_lengths[i]] for i in range(batch_size)]
    phoneme_target = [phoneme_target[i][:phoneme_target_lengths[i]] for i in range(batch_size)]

    # ── Phase 1: build all cost matrices on the GPU ─────────────────────
    # (no CPU syncs during this loop — every op stays on device)
    prepared = []
    costs_list = []
    for pred, y, pred_p, y_p in zip(predictions, ground_truth, pred_phone, phoneme_target):
        y = y.to(device)
        pred = pred.to(device)
        pred_p = pred_p.to(device)
        y_p = y_p.to(device)

        # ── (1) Augment with temporal deltas ───────────────────────────
        y_aug = _append_deltas(y)
        pred_aug = _append_deltas(pred)

        # ── (2) Apply band weights (repeat 3× for mel/∆/∆² blocks) ────
        if use_band_weight:
            aug_w = band_w.repeat_interleave(3).sqrt()
            y_aug = y_aug * aug_w
            pred_aug = pred_aug * aug_w

        # ── (2) Cost matrix: L1 or L2 ──────────────────────────────────
        if use_l1_cost:
            costs = _l1_cost_matrix(y_aug, pred_aug)
        else:
            dists = cdist(y_aug.unsqueeze(0), pred_aug.unsqueeze(0), p=2)
            costs = dists.squeeze(0)

        costs_list.append(costs)
        prepared.append((pred, y, pred_p, y_p))

    # ── Phase 2: DTW alignments on CPU — one per sample, in parallel ──
    # `time_warp` is @jit(nogil=True), so the O(T²) DP releases the GIL and multiple samples' alignments run concurrently on the CPU.
    alignments = list(_ALIGN_EXECUTOR.map(_align_from_costs, costs_list))

    # ── Phase 3: losses on the GPU (using the precomputed alignments) ──
    losses = []
    correct_phones = 0.0
    total_length = 0.0
    phoneme_loss = torch.tensor(0.0, device=device)

    for (pred, y, pred_p, y_p), costs, alignment in zip(prepared, costs_list, alignments):
        # ── Core DTW path cost ─────────────────────────────────────────
        loss = costs[alignment, range(len(alignment))].sum()

        # ── (5) Post-alignment L1 on aligned frame pairs ──────────────
        if use_post_alignment_l1 and post_alignment_weight > 0:
            aligned_gt = y[alignment]  # GT warped to prediction timeline
            post_l1 = l1_loss(pred, aligned_gt, reduction='sum')
            loss = loss + post_alignment_weight * post_l1

        # ── (4) Spectral contrast loss ─────────────────────────────────
        if use_contrast_loss and contrast_weight > 0:
            aligned_gt = y[alignment]
            pred_range = pred.max(-1)[0] - pred.min(-1)[0]
            gt_range = aligned_gt.max(-1)[0] - aligned_gt.min(-1)[0]
            contrast_penalty = l1_loss(pred_range, gt_range, reduction='sum')
            loss = loss + contrast_weight * contrast_penalty

        losses.append(loss)
        total_length += pred.size(0)

        # ── Phoneme loss (unchanged) ───────────────────────────────────
        y_p = pad_or_trim_sequence(y_p, y.size(0), pad_value=y_p[-1].item() if y_p.numel() else 0)
        alignment_idx = tensor(alignment, dtype=torch.long, device=device)
        y_p_aligned = y_p[alignment_idx]
        if pred_p.size(0) != y_p_aligned.size(0):
            print(f"Warning: phoneme prediction length {pred_p.size(0)} does not match aligned target length {y_p_aligned.size(0)}, skipping phoneme loss for this sample")
            continue
        phoneme_loss = phoneme_loss + cross_entropy(pred_p, y_p_aligned, reduction='sum')
        correct_phones += (pred_p.argmax(dim=-1) == y_p_aligned).sum().item()

    mel_loss = (sum(losses) / batch_size) / total_length
    phoneme_loss = (phoneme_loss / batch_size) / total_length
    return mel_loss, phoneme_loss, correct_phones / total_length, alignments