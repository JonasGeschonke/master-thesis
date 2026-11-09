import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

class MultiScaleMelLoss(Module):
    """Zero-parameter multi-scale mel spectrogram loss."""

    def __init__(self, scales: list[float] | None = None, n_mels: int = 128):
        super().__init__()
        self.scales = scales or [0.0, 1.0, 2.0, 4.0]
        self.n_mels = n_mels

        self._kernels: dict[float, Tensor | None] = {}
        for sigma in self.scales:
            if sigma <= 0.0:
                self._kernels[sigma] = None
            else:
                radius = int(3.0 * sigma)
                t = torch.arange(-radius, radius + 1, dtype=torch.float32)
                k = torch.exp(-0.5 * (t / sigma) ** 2)
                k = k / k.sum()
                self._kernels[sigma] = k.view(1, 1, -1)

    def _blur(self, mel: Tensor, sigma: float) -> Tensor:
        """Apply Gaussian blur along the frequency axis (last dim)."""
        kernel = self._kernels.get(sigma)
        if kernel is None:
            return mel

        kernel = kernel.to(mel.device, dtype=mel.dtype)

        squeeze_batch = mel.dim() == 2
        if squeeze_batch:
            mel = mel.unsqueeze(0)

        B, T, n_freq = mel.shape
        pad = kernel.shape[-1] // 2
        x = mel.reshape(B * T, 1, n_freq)
        x = F.conv1d(x, kernel, groups=1, padding=pad)
        x = x.reshape(B, T, n_freq)

        if squeeze_batch:
            x = x.squeeze(0)
        return x

    def forward(self, pred_mel: Tensor, gt_mel: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Compute multi-scale spectral convergence and log-magnitude loss."""
        squeeze_batch = pred_mel.dim() == 2
        if squeeze_batch:
            pred_mel = pred_mel.unsqueeze(0)
            gt_mel = gt_mel.unsqueeze(0)

        if mask is None:
            mask_f: Tensor | None = None
        else:
            mask_f = mask.unsqueeze(-1).to(dtype=pred_mel.dtype)

        sc_total = torch.tensor(0.0, device=pred_mel.device)
        mag_total = torch.tensor(0.0, device=pred_mel.device)
        n_scales = len(self.scales)

        for sigma in self.scales:
            p = self._blur(pred_mel, sigma)
            g = self._blur(gt_mel, sigma)

            if mask_f is not None:
                p = p * mask_f
                g = g * mask_f
                # Per-sample spectral convergence over valid frames
                sc_num = torch.linalg.vector_norm(g - p, dim=(1, 2))
                sc_den = torch.linalg.vector_norm(g, dim=(1, 2)) + 1e-8
                sc_total = sc_total + (sc_num / sc_den).mean()

                # Log-magnitude L1 over valid frames, per sample then averaged
                log_p = torch.log(p.clamp(min=1e-7))
                log_g = torch.log(g.clamp(min=1e-7))
                diff = (log_p - log_g).abs() * mask_f
                # valid elements = valid frames × n_mels (mask last dim is 1)
                counts = mask_f.sum(dim=(1, 2)) * pred_mel.size(-1)
                mag_total = mag_total + (diff.sum(dim=(1, 2)) / counts.clamp(min=1)).mean()
            else:
                # Single (or fully-valid) batch — original global reduction
                denom = torch.norm(g, p='fro') + 1e-8
                sc_total = sc_total + torch.norm(g - p, p='fro') / denom

                # Log-magnitude L1
                log_p = torch.log(p.clamp(min=1e-7))
                log_g = torch.log(g.clamp(min=1e-7))
                mag_total = mag_total + F.l1_loss(log_p, log_g)

        return sc_total / n_scales, mag_total / n_scales