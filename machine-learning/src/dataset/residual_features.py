import librosa
import numpy as np
import soundfile as sf

from __future__ import annotations
from scipy.signal import butter, sosfilt

class ResidualFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 40,
        n_mfcc: int = 13,
        subband_centers: list[float] | None = None,
    ):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        if subband_centers is None:
            self.subband_centers = [
                250,     # Low-frequency breath / nasal murmur
                500,     # Low breath
                1000,    # Lower formant region
                2000,    # /ʃ/ lower edge
                3000,    # /ʃ/ primary
                4000,    # General frication
                5000,    # /f/ region
                6300,    # Between /f/ and /s/
                8000,    # /s/ primary peak
                10000,   # /s/ upper
                12000,   # High frication
                14000,   # Very high
                16000,   # Near-ultrasound band
                18000,   # Below 19 kHz carrier
            ]
        self.n_subbands = len(self.subband_centers)

        self._freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

    def forward(self, waveform: np.ndarray) -> np.ndarray:
        # Compute all Phase-1 residual features from a raw audio waveform.
        subband = self._subband_energies(waveform)
        flux = self._spectral_flux(waveform)
        moments = self._spectral_moments(waveform)
        zcr = self._zero_crossing_rate(waveform)
        flatness = self._spectral_flatness(waveform)
        mfcc = self._residual_mfcc(waveform)
        teo = self._teager_energy(waveform)

        return np.concatenate([subband, flux, moments, zcr, flatness, mfcc, teo], axis=1)  # [T, 34]

    @property
    def n_features(self) -> int:
        """Total number of Phase-1 feature dimensions."""
        return 34

    def get_feature_names(self) -> list[str]:
        """Return human-readable names for each feature dimension (for debugging/plotting)."""
        names = []
        for i, cf in enumerate(self.subband_centers):
            names.append(f"subband_{int(cf)}Hz")
        names += ["spectral_flux"]
        names += ["centroid", "spread", "skewness"]
        names += ["zcr"]
        names += ["spectral_flatness"]
        for i in range(1, self.n_mfcc + 1):
            names.append(f"mfcc_{i}")
        names += ["teager_energy"]
        return names

    # ──────────────────────────────────────────────────────────────────────
    #  Individual feature methods
    # ──────────────────────────────────────────────────────────────────────

    def _subband_energies(self, waveform: np.ndarray) -> np.ndarray:
        """Compute log-energy in consonant-relevant frequency subbands."""

        # Determine the number of STFT frames by computing a quick STFT.
        n_frames = librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=True).shape[1]
        energies = np.zeros((n_frames, self.n_subbands), dtype=np.float32)

        for i, fc in enumerate(self.subband_centers):
            # Design a bandpass filter with ~1/3 octave bandwidth.
            bw = fc * 0.3
            low = max(1.0, fc - bw / 2)
            high = min(self.sr / 2 - 1.0, fc + bw / 2)
            if low >= high:
                energies[:, i] = 0.0
                continue
            sos = butter(2, [low, high], btype="band", fs=self.sr, output="sos")
            filtered = sosfilt(sos, waveform)

            # Frame energy using STFT-style framing (centre-padded).
            pad_len = self.win_length // 2
            # Use np.pad to centre-pad, then frame with hop_length.
            padded = np.pad(filtered, pad_len, mode="reflect")
            # Need at least win_length samples for one frame
            if len(padded) < self.win_length:
                padded = np.pad(padded, (0, self.win_length - len(padded)), mode="constant")
            framed = librosa.util.frame(padded, frame_length=self.win_length, hop_length=self.hop_length)
            rms = np.sqrt(np.mean(framed**2, axis=0) + 1e-12)
            # Interpolate to exactly n_frames (handles off-by-one from edge effects)
            x_old = np.linspace(0, 1, len(rms))
            x_new = np.linspace(0, 1, n_frames)
            rms_aligned = np.interp(x_new, x_old, rms)
            energies[:, i] = 10.0 * np.log10(rms_aligned + 1e-12)

        return energies

    def _spectral_flux(self, waveform: np.ndarray) -> np.ndarray:
        """Compute the L2 norm of the frame-to-frame log-magnitude spectrum difference."""
        S = np.abs(librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length))
        log_S = np.log(S + 1e-10)
        diff = np.diff(log_S, axis=1)
        flux = np.sqrt(np.sum(diff**2, axis=0))

        # Prepend the first value so shape matches [T].
        flux = np.concatenate([[flux[0]], flux])

        # Smooth with a 3-frame moving average to suppress noise spikes.
        flux = np.convolve(flux, [1.0 / 3] * 3, mode="same")
        return flux[:, np.newaxis]

    def _spectral_moments(self, waveform: np.ndarray) -> np.ndarray:
        """Compute spectral centroid, spread, and skewness per frame."""
        S = np.abs(librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length))
        freqs = self._freqs[:, np.newaxis]

        total_energy = np.sum(S, axis=0) + 1e-10

        # Centroid = Σ(f * S(f)) / Σ(S(f))
        centroid = np.sum(freqs * S, axis=0) / total_energy

        # Spread = sqrt(Σ((f - centroid)² * S(f)) / Σ(S(f)))
        diff = freqs - centroid[np.newaxis, :]
        spread = np.sqrt(np.sum(diff**2 * S, axis=0) / total_energy)

        # Skewness = Σ((f - centroid)³ * S(f)) / (spread³ * Σ(S(f)))
        skewness = np.sum(diff**3 * S, axis=0) / (spread**3 * total_energy + 1e-10)
        out = np.stack([centroid, spread, skewness], axis=1)
        # Replace any inf/nan with 0.
        out[~np.isfinite(out)] = 0.0
        return out

    def _zero_crossing_rate(self, waveform: np.ndarray) -> np.ndarray:
        """Compute the zero-crossing rate per frame."""
        zcr = librosa.feature.zero_crossing_rate(
            waveform, frame_length=self.win_length, hop_length=self.hop_length, center=True
        ).T
        return zcr.astype(np.float32)

    def _spectral_flatness(self, waveform: np.ndarray) -> np.ndarray:
        """Compute spectral flatness (Wiener entropy) per frame."""
        S = np.abs(librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length))
        S = S + 1e-10
        geom = np.exp(np.mean(np.log(S), axis=0))
        arith = np.mean(S, axis=0)
        flatness = geom / arith
        flatness[~np.isfinite(flatness)] = 0.0
        return flatness[:, np.newaxis]

    def _residual_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """Compute the lower MFCC coefficients (excluding C0)."""
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sr,
            n_mfcc=self.n_mfcc + 1,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            center=True,
        ).T
        return mfcc[:, 1:].astype(np.float32)

    def _teager_energy(self, waveform: np.ndarray) -> np.ndarray:
        """Compute frame-level Teager Energy Operator (TEO)."""
        # Compute TEO on the raw waveform (unframed).
        teo = waveform[1:-1] ** 2 - waveform[:-2] * waveform[2:]

        # Pad edges by replicating the first/last value.
        teo = np.concatenate([[teo[0]], teo, [teo[-1]]])

        # Frame the TEO signal using the same STFT frame grid.
        pad_len = self.win_length // 2
        teo_padded = np.pad(teo, pad_len, mode="reflect")
        if len(teo_padded) < self.win_length:
            teo_padded = np.pad(teo_padded, (0, self.win_length - len(teo_padded)), mode="constant")
        framed = librosa.util.frame(teo_padded, frame_length=self.win_length, hop_length=self.hop_length)
        frame_teo = np.mean(framed, axis=0)  # [T_raw]

        # Interpolate to the canonical STFT frame count.
        n_frames = librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=True).shape[1]
        x_old = np.linspace(0, 1, len(frame_teo))
        x_new = np.linspace(0, 1, n_frames)
        frame_teo_aligned = np.interp(x_new, x_old, frame_teo)
        return frame_teo_aligned[:, np.newaxis].astype(np.float32)  # [T, 1]

    @staticmethod
    def zscore_per_feature(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Z-score normalise each feature dimension independently."""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + eps
        return (features - mean) / std

def load_residual_features(
    filename: str,
    extractor: ResidualFeatureExtractor | None = None,
    target_sr: int = 44100,
    normalize: bool = False,
) -> np.ndarray:
    """Load audio and extract residual features in one call."""
    if extractor is None:
        extractor = ResidualFeatureExtractor(sample_rate=target_sr)

    audio, orig_sr = sf.read(filename)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample to target sample rate.
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    features = extractor.forward(audio)  # [T, 34]

    if normalize:
        features = ResidualFeatureExtractor.zscore_per_feature(features)

    return features
