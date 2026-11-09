import random
import torch
import numpy as np

from pathlib import Path
from torch import Tensor
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from utils.utils_transformer import PHONEME_INVENTORY

ULTRASR_FEATURE_NAMES = ['|S| (Magnitude)', 'Φ (Phase)', 'ΔΦ (Delta Phase)',
                         'Δ²Φ (Double-Delta)', '|ΔS| (Doppler)']
ULTRASR_CMAPS = ['magma', 'twilight', 'RdBu', 'RdBu', 'magma']

def plot_ultrasr_features(ultrasr: Tensor, save_path: Path | None = None,
                          max_time_frames: int | None = None) -> plt.Figure:
    """Plot 5-row heatmap of UltraSR features [T, n_bins, 5]."""
    if ultrasr.dim() == 4:
        ultrasr = ultrasr[0]
    data = ultrasr.cpu().numpy() if ultrasr.is_cuda else ultrasr.numpy()
    if max_time_frames is not None:
        data = data[:max_time_frames]
    T, n_bins, _ = data.shape
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    for i in range(5):
        im = axes[i].imshow(data[:, :, i].T, aspect='auto', origin='lower',
                             cmap=ULTRASR_CMAPS[i], interpolation='nearest')
        axes[i].set_ylabel(ULTRASR_FEATURE_NAMES[i])
        plt.colorbar(im, ax=axes[i], fraction=0.05)
    axes[-1].set_xlabel('Time frame')
    fig.suptitle('UltraSR Broadband Ultrasound Reflection Features')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    return fig

def show_spectrograms_transformer(model, random_batch, checkpoint, image_dir,
                                  hyperparameters, forcing_ratio):
    use_voiced = random.random() > forcing_ratio
    source_key = 'voiced'
    lp_sequence = random_batch[source_key]['lp']
    pha_sequence = random_batch[source_key]['pha']
    amp_sequence = random_batch[source_key]['amp']
    dop_sequence = random_batch[source_key]['dop']
    lengths = random_batch[source_key]['lengths']
    lp_sequence = lp_sequence.to(hyperparameters.device, non_blocking=True)
    pha_sequence = pha_sequence.to(hyperparameters.device, non_blocking=True)
    amp_sequence = amp_sequence.to(hyperparameters.device, non_blocking=True)
    dop_sequence = dop_sequence.to(hyperparameters.device, non_blocking=True)
    lengths = lengths.to(hyperparameters.device, non_blocking=True)

    with torch.no_grad():
        output_seq, output_lengths, _ = model(
            lp_sequence, pha_sequence, amp_sequence, dop_sequence,
            lengths=lengths,
        )

    output_seq = output_seq.cpu().numpy()
    output_lengths = output_lengths.cpu().numpy()
    gt_seq = random_batch['gt']['data'].cpu().numpy()
    gt_lengths = random_batch['gt']['lengths'].cpu().numpy()
    lp_np = lp_sequence.cpu().numpy()
    pha_np = pha_sequence.cpu().numpy()
    amp_np = amp_sequence.cpu().numpy()
    dop_np = dop_sequence.cpu().numpy()

    for i in range(len(output_seq)):
        output_spec = output_seq[i][:output_lengths[i]]
        fig, ax = plt.subplots(3, 2, figsize=(14, 10))
        ax[0, 0].imshow(output_spec.T, aspect='auto', origin='lower')
        ax[0, 0].set_title(f"Reconstructed Spectrogram - Checkpoint: {checkpoint}")
        ax[0, 1].imshow(gt_seq[i][:gt_lengths[i]].T, aspect='auto', origin='lower')
        ax[0, 1].set_title("Original Spectrogram")
        ax[1, 0].imshow(lp_np[i, :lengths[i]].T, aspect='auto', origin='lower')
        ax[1, 0].set_title("Low-pass Input Spectrogram")

        # UltraSR Doppler spectrogram (1024 bins, full FFT)
        dop_slice = dop_np[i, :lengths[i]].T
        ax[1, 1].imshow(dop_slice, aspect='auto', origin='lower', cmap='viridis')
        ax[1, 1].set_title("UltraSR Doppler Spectrogram (1024 bins)")

        # UltraSR amplitude & phase gradients
        T_vis = min(lengths[i].item(), dop_np.shape[1])
        t_axis = np.arange(T_vis)
        amp_grad = amp_np[i, :T_vis, 0]
        pha_grad = pha_np[i, :T_vis, 0]
        ax[2, 0].plot(t_axis, amp_grad, color='steelblue', label='Amp Grad')
        ax[2, 0].set_title("UltraSR Amplitude Gradient")
        ax[2, 0].set_xlabel('Time frame')
        ax[2, 1].plot(t_axis, pha_grad, color='darkorange', label='Pha Grad')
        ax[2, 1].set_title("UltraSR Phase Gradient")
        ax[2, 1].set_xlabel('Time frame')

        plt.tight_layout()
        artifact_path = f"{checkpoint}_sample_{i}.png"
        plt.savefig(image_dir / artifact_path)
        plt.close()

def plot_spectrogram_with_phoneme_timeline(random_batch, sample_index: int = 0, phoneme_inventory: list[str] = PHONEME_INVENTORY, sample_rate: int = 22050, hop_length: int = 256, title: str | None = None, save_path: Path | None = None):
    """Plot the first sample spectrogram with its aligned phoneme timeline."""
    voiced_spec = random_batch['voiced']['lp'][sample_index].cpu().numpy()
    voiced_length = int(random_batch['voiced']['lengths'][sample_index].item())
    # Doppler spectrogram (1024 bins, full FFT) from separate UltraSR views
    voiced_dop = random_batch['voiced']['dop'][sample_index].cpu().numpy()
    voiced_dop_length = int(random_batch['voiced']['lengths'][sample_index].item())
    phoneme_seq = random_batch['phoneme']['data'][sample_index].cpu().numpy()
    phoneme_length = int(random_batch['phoneme']['lengths'][sample_index].item())

    voiced_spec = voiced_spec[:voiced_length]
    phoneme_seq = phoneme_seq[:phoneme_length]
    # UltraSR Doppler spectrogram [T, 1024]
    doppler_spec = voiced_dop[:voiced_dop_length]
    num_frames = voiced_spec.shape[0]

    fig, (ax_dop, ax_spec, ax_phoneme) = plt.subplots(3, 1, figsize=(16, 7), gridspec_kw={'height_ratios': [4, 4, 1]}, sharex=True)
    ax_dop.imshow(doppler_spec.T, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax_dop.set_ylabel('Doppler bins')
    ax_dop.set_title(title or 'Spectrogram with aligned phoneme sequence')
    im = ax_spec.imshow(voiced_spec.T, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
    ax_spec.set_ylabel('Frequency bins')

    boundaries = []
    segments = []
    if phoneme_seq.size > 0:
        current_label = phoneme_seq[0]
        start = 0
        for frame_index, label in enumerate(phoneme_seq[1:], start=1):
            if label != current_label:
                segments.append((start, frame_index, int(current_label)))
                boundaries.append(frame_index)
                start = frame_index
                current_label = label
        segments.append((start, len(phoneme_seq), int(current_label)))
        boundaries.append(len(phoneme_seq))

    cmap = plt.get_cmap('tab20', max(2, len(phoneme_inventory)))
    ax_phoneme.set_ylim(0, 1)
    ax_phoneme.set_yticks([])
    ax_phoneme.set_xlim(0, num_frames)
    ax_phoneme.set_xlabel('Frame index')

    for start, end, label in segments:
        color = cmap(label % cmap.N)
        rect = Rectangle((start, 0), end - start, 1, facecolor=color, edgecolor='none', alpha=0.75)
        ax_phoneme.add_patch(rect)
        if end - start >= 6:
            label_text = phoneme_inventory[label] if 0 <= label < len(phoneme_inventory) else 'UNK'
            ax_phoneme.text((start + end) / 2, 0.5, label_text, ha='center', va='center', fontsize=8, color='black', clip_on=True)

    for boundary in boundaries:
        ax_spec.axvline(boundary, color='white', alpha=0.25, linewidth=0.5)
        ax_phoneme.axvline(boundary, color='black', alpha=0.15, linewidth=0.5)

    ax_spec.set_xticks(np.linspace(0, num_frames, 9))
    time_labels = [f'{(float(x) * hop_length / sample_rate):.2f}s' for x in np.linspace(0, num_frames, 9)]
    ax_phoneme.set_xticklabels(time_labels)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return fig, (ax_spec, ax_phoneme)

def plot_dtw_alignment(costs: Tensor, title: str | None = None, save_path: Path | None = None):
    costs_np = costs.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(costs_np.T, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xlabel('Predicted sequence index')
    ax.set_ylabel('Ground truth sequence index')
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label='DTW Cost')
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()