import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal as scipy_signal

class UltraSRFeatureExtractor(nn.Module):
    def __init__(self, fs=44100, fc=19000, stft_n_fft=1024, stft_hop=256, lpf_cutoff=800.0):
        super().__init__()
        self.fs = fs
        self.fc = fc
        self.n_fft = stft_n_fft
        self.hop_length = stft_hop
        self.lpf_cutoff = lpf_cutoff

    @staticmethod
    def detect_carrier_frequency(
        waveform: np.ndarray,
        sr: int,
        search_range: tuple[float, float] = (18000.0, 20000.0),
        min_snr: float = 3.0,
    ) -> tuple[float, float]:
        """Detect the strongest sinusoidal carrier in the 18-20 kHz band."""
        f, psd = scipy_signal.welch(
            waveform, sr, nperseg=min(4096, len(waveform) // 4),
            return_onesided=True,
        )
        mask = (f >= search_range[0]) & (f <= search_range[1])
        if not mask.any():
            fallback = (search_range[0] + search_range[1]) / 2.0
            return fallback, 0.0

        band_psd = psd[mask]
        band_freqs = f[mask]
        peak_idx = int(np.argmax(band_psd))
        peak_freq = float(band_freqs[peak_idx])
        peak_power = band_psd[peak_idx]

        # Noise floor: median of band excluding ±100 Hz around the peak
        exclude_mask = np.abs(band_freqs - peak_freq) > 100.0
        if np.any(exclude_mask):
            noise_floor = float(np.median(band_psd[exclude_mask]))
        else:
            noise_floor = float(np.median(band_psd))

        snr_db = 10.0 * np.log10((peak_power + 1e-20) / (noise_floor + 1e-20))

        if snr_db < min_snr:
            # Peak too weak — use fallback
            fallback = (search_range[0] + search_range[1]) / 2.0
            return fallback, snr_db

        return peak_freq, snr_db

    def apply_lowpass(self, waveform):
        return torchaudio.functional.lowpass_biquad(waveform, self.fs, self.lpf_cutoff)

    @staticmethod
    def _unwrap_phase(phase_wrapped: torch.Tensor) -> torch.Tensor:
        """Unwrap phase along the last dimension (standard 2π-jump correction)."""
        phase = phase_wrapped.clone()
        dd = torch.diff(phase_wrapped, dim=-1)
        ddmod = torch.remainder(dd + torch.pi, 2 * torch.pi) - torch.pi
        ddmod[ddmod == -torch.pi] = torch.pi
        ph_correct = ddmod - dd
        ph_correct[dd.abs() < torch.pi] = 0.0
        phase[..., 1:] = phase[..., 1:] + torch.cumsum(ph_correct, dim=-1)
        return phase

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Global z-score normalisation per sample (across all elements)."""
        m = x.mean()
        s = x.std() + eps
        return (x - m) / s

    def forward(self, x, fc: float | None = None, gain: float = 1.0):
        fc = self.fc if fc is None else fc
        T = x.size(0)
        t = torch.arange(T, device=x.device).float() / self.fs

        # ---- 1. Coherent Demodulation (at detected carrier frequency) ----
        cos_carrier = torch.cos(2 * torch.pi * fc * t).unsqueeze(0)
        sin_carrier = -torch.sin(2 * torch.pi * fc * t).unsqueeze(0)

        mixed_i = x * cos_carrier
        mixed_q = x * sin_carrier

        i_base = self.apply_lowpass(mixed_i)
        q_base = self.apply_lowpass(mixed_q)

        # Optional gain boost to lift weak baseband signals above numerical noise
        if gain != 1.0:
            i_base = i_base * gain
            q_base = q_base * gain

        # ---- 2. Envelope and Phase from I/Q ----
        amplitude = torch.sqrt(i_base**2 + q_base**2 + 1e-8)
        phase_wrapped = torch.atan2(q_base, i_base)

        # Light smoothing
        amplitude = F.avg_pool1d(
            amplitude.unsqueeze(1), kernel_size=5, stride=1, padding=2
        ).squeeze(1)
        phase_wrapped = F.avg_pool1d(
            phase_wrapped.unsqueeze(1), kernel_size=5, stride=1, padding=2
        ).squeeze(1)

        # ---- 3. Phase Unwrapping (critical for meaningful gradients) ----
        phase = self._unwrap_phase(phase_wrapped)

        # ---- 4. View of Distance & Energy (gradients = velocity) ----
        amp_grad = torch.diff(amplitude, dim=-1, prepend=amplitude[:, :1])
        pha_grad = torch.diff(phase, dim=-1, prepend=phase[:, :1])

        # Downsample to hop-rate grid, matching STFT frame count (center=True → T//hop + 1 frames).
        n_stft_frames = T // self.hop_length + 1
        amp_grad_down = F.interpolate(
            amp_grad.unsqueeze(0), size=n_stft_frames, mode='linear', align_corners=False
        ).squeeze(0)
        pha_grad_down = F.interpolate(
            pha_grad.unsqueeze(0), size=n_stft_frames, mode='linear', align_corners=False
        ).squeeze(0)

        # ---- 5. View of Speed (Doppler Spectrogram) ----
        complex_baseband = torch.complex(i_base, q_base)
        stft_out = torch.stft(
            complex_baseband,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            return_complex=True,
            window=torch.hann_window(self.n_fft, device=x.device),
        )

        doppler_mag = torch.abs(stft_out) ** 2

        # Suppress residual carrier leakage.
        doppler_mag[:, 0:3, :] = 1e-8
        doppler_mag[:, -2:, :] = 1e-8

        doppler_spec = 10 * torch.log10(doppler_mag + 1e-8)

        # ---- 6. Temporal Alignment ----
        min_time = min(amp_grad_down.shape[-1], doppler_spec.shape[-1])
        amp_grad_down = amp_grad_down[:, :min_time].unsqueeze(1)
        pha_grad_down = pha_grad_down[:, :min_time].unsqueeze(1)
        doppler_spec = doppler_spec[:, :, :min_time]

        # ---- 7. Per-sample z-score normalisation ----
        amp_grad_norm = self._zscore(amp_grad_down)
        pha_grad_norm = self._zscore(pha_grad_down)
        doppler_norm = self._zscore(doppler_spec)

        return pha_grad_norm, amp_grad_norm, doppler_norm
    
def _resample_to_target_sr(audio: np.ndarray, orig_sr: int, target_sr: int = 44100) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def visualize_features(raw_signal, fs, title="Ultrasound Features"):
    extractor = UltraSRFeatureExtractor(fs=fs, fc=19000)
    sig_tensor = torch.tensor(raw_signal).unsqueeze(0).float()
    with torch.no_grad():
        pha_grad, amp_grad, doppler = extractor(sig_tensor)
    
    p_grad = pha_grad.squeeze().numpy()
    a_grad = amp_grad.squeeze().numpy()
    dop_spec = doppler.squeeze().numpy()
    
    time_axis = np.linspace(0, len(raw_signal)/fs, len(p_grad))
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # 1. Raw / Demodulated approximation (or reference audio)
    axs[0].plot(np.linspace(0, len(raw_signal)/fs, len(raw_signal)), raw_signal, color='blue')
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Received Signal")
    
    # 2. Amplitude Gradient
    axs[1].plot(time_axis, a_grad, color='steelblue')
    axs[1].set_ylabel("Amplitude Grad")
    
    # 3. Phase Gradient
    axs[2].plot(time_axis, p_grad, color='darkorange')
    axs[2].set_ylabel("Phase Grad")
    
    # 4. Doppler Spectrogram
    # The paper plots it around the center frequency mapping
    freq_bins = np.linspace(-fs/2, fs/2, dop_spec.shape[0]) 
    im = axs[3].pcolormesh(time_axis, freq_bins, dop_spec, shading='gouraud', cmap='viridis')
    axs[3].set_ylabel("Frequency (Hz)")
    axs[3].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axs[3], label="dB")
    
    plt.tight_layout()
    plt.show()

def load_ultrasr_features(
    filename,
    extractor: UltraSRFeatureExtractor,
    auto_detect_carrier: bool = True,
    gain: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Load audio, detect the carrier frequency, and extract UltraSR features."""
    audio, r = sf.read(filename)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = _resample_to_target_sr(audio, r, extractor.fs)

    # Detect actual carrier frequency in the resampled audio
    carrier_freq = float(extractor.fc)  # fallback to constructor default
    carrier_snr = 0.0
    if auto_detect_carrier:
        detected_freq, detected_snr = extractor.detect_carrier_frequency(
            audio, extractor.fs
        )
        carrier_freq = detected_freq
        carrier_snr = detected_snr

    audio_tensor = torch.tensor(audio, dtype=torch.float32, device='cpu')
    pha_grad, amp_grad, doppler_spec = extractor(audio_tensor, fc=carrier_freq, gain=gain)
    return pha_grad, amp_grad, doppler_spec, carrier_freq, carrier_snr