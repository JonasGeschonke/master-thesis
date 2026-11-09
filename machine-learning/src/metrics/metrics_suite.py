from __future__ import annotations

import time
import warnings
import bigvgan
import whisper
import numpy as np

from typing import Optional, List, Dict, Any
from torch import Tensor, no_grad, inference_mode
from torch.nn import Module
from torch.utils.data import DataLoader

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
    compute_waveform_mse,
)
from metrics.intelligibility import (
    compute_per,
    compute_phoneme_confusion_matrix,
    compute_wer_cer,
)
from metrics.ssi_specific import (
    compute_dtw_alignment_cost,
    compute_per_phoneme_metrics,
    compute_per_phoneme_stoi,
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


def _vocode_to_waveform(
    mel: Tensor,
    vocoder: Optional[Module],
    hyperparameters,
    mel_length: int,
) -> Optional[np.ndarray]:
    """Convert a mel spectrogram to waveform via BigVGAN vocoder."""
    if vocoder is None:
        return None

    device = getattr(hyperparameters, 'device', 'cpu')

    # Prepare mel: [1, n_mels, T]
    mel_input = mel[:mel_length].unsqueeze(0).permute(0, 2, 1).to(device)

    with inference_mode():
        try:
            wav = vocoder(mel_input)  # [1, 1, T_audio]
            wav = wav.squeeze().cpu().numpy()
            return wav
        except Exception as e:
            warnings.warn(f"Vocoding failed: {e}")
            return None


def _load_vocoder(hyperparameters):
    """Lazy-load BigVGAN vocoder."""
    try:
        device = getattr(hyperparameters, 'device', 'cpu')
        loader_kwargs = dict(
            use_cuda_kernel=False,
        )
        try:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_44khz_128band_256x",
                proxies=None,
                resume_download=False,
                **loader_kwargs,
            )
        except TypeError:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_44khz_128band_256x",
                **loader_kwargs,
            )
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
        return vocoder
    except ImportError:
        warnings.warn("bigvgan not installed; waveform metrics unavailable.")
        return None
    except Exception as e:
        warnings.warn(f"Failed to load BigVGAN: {e}")
        return None


class MetricsSuite:
    """Comprehensive evaluation suite for SSI reconstruction models."""

    def __init__(
        self,
        model: Module,
        hyperparameters,
        phoneme_inventory: Optional[List[str]] = None,
        vocoder: Optional[Module] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.model = model
        self.hyperparameters = hyperparameters
        self.phoneme_inventory = phoneme_inventory or []
        self.vocoder = vocoder
        self.checkpoint_path = checkpoint_path
        self.device = getattr(hyperparameters, 'device', 'cpu')

    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        compute_waveform_metrics: bool = True,
        compute_efficiency: bool = True,
        vocode_samples: int = 10,
        training_history: Optional[Dict[str, Any]] = None,
        train_on: str = "voiced",
    ) -> Dict[str, Any]:
        """Run full evaluation across all metric groups."""
        model = self.model.to(self.device)
        model.eval()

        # Ensure vocoder is loaded if waveform metrics requested
        if compute_waveform_metrics and self.vocoder is None:
            self.vocoder = _load_vocoder(self.hyperparameters)

        # Accumulators for batch-level metrics
        accum = self._init_accumulators()

        samples_vocoded = 0
        all_per_phoneme: Dict[str, List[float]] = {}
        all_confusion = None

        start_time = time.perf_counter()

        with no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                src_key = train_on if train_on in (
                    'voiced', 'unvoiced') else 'voiced'

                # --- Unpack batch ---
                lp_seq = batch[src_key]['lp'].to(self.device)
                pha_seq = batch[src_key]['pha'].to(self.device)
                amp_seq = batch[src_key]['amp'].to(self.device)
                dop_seq = batch[src_key]['dop'].to(self.device)
                lengths = batch[src_key]['lengths'].to(self.device)

                gt_mel = batch['gt']['data'].to(self.device)
                gt_lengths = batch['gt']['lengths'].to(self.device)
                phoneme_gt = batch['phoneme']['data'].to(self.device)
                phoneme_lengths = batch['phoneme']['lengths'].to(self.device)
                transcripts = batch.get('transcript', [])

                B = lp_seq.size(0)

                # --- Forward pass ---
                mel_out, out_lengths, phoneme_logits = model(
                    lp_seq, pha_seq, amp_seq, dop_seq, lengths=lengths,
                )

                # --- Per-sample metrics ---
                for b in range(B):
                    pred_len = int(out_lengths[b].item())
                    gt_len = int(gt_lengths[b].item())
                    if pred_len == 0 or gt_len == 0:
                        continue

                    pred_mel = mel_out[b, :pred_len, :]
                    gt_mel_sample = gt_mel[b, :gt_len, :]
                    phon_len = int(phoneme_lengths[b].item())

                    # -- Group 1: Spectral --
                    try:
                        mcd_v = compute_mcd(pred_mel, gt_mel_sample)
                        lsd_v = compute_lsd(pred_mel, gt_mel_sample)
                        rmse_v = compute_mel_rmse(pred_mel, gt_mel_sample)
                        pearson = compute_pearson_per_band(
                            pred_mel, gt_mel_sample)
                    except Exception:
                        mcd_v = lsd_v = rmse_v = float("nan")
                        pearson = np.array([float("nan")])

                    accum["mcd"].append(mcd_v)
                    accum["lsd"].append(lsd_v)
                    accum["mel_rmse"].append(rmse_v)
                    pearson_arr = np.asarray(pearson, dtype=float).ravel()
                    finite_pearson = pearson_arr[np.isfinite(pearson_arr)]
                    accum["pearson_mean"].append(
                        float(np.mean(finite_pearson)) if finite_pearson.size > 0 else float("nan"))

                    # -- Group 4: DTW cost --
                    try:
                        dtw = compute_dtw_alignment_cost(
                            pred_mel, gt_mel_sample)
                    except Exception:
                        dtw = float("nan")
                    accum["dtw_cost"].append(dtw)

                    # -- Group 3: PER and confusion --
                    if phon_len > 0:
                        pred_phon = phoneme_logits[b,
                                                   :phon_len, :]  # [T_phon, V]
                        gt_phon = phoneme_gt[b, :phon_len]
                        try:
                            per_val = compute_per(
                                pred_phon.unsqueeze(0), gt_phon.unsqueeze(0),
                                phoneme_inventory=self.phoneme_inventory,
                            )
                        except Exception:
                            per_val = float("nan")
                        accum["per"].append(per_val)

                        # Accumulate confusion matrix
                        try:
                            cm, _ = compute_phoneme_confusion_matrix(
                                pred_phon.unsqueeze(0), gt_phon.unsqueeze(0),
                                num_classes=len(
                                    self.phoneme_inventory) or pred_phon.size(-1),
                            )
                            if all_confusion is None:
                                all_confusion = cm
                            else:
                                all_confusion += cm
                        except Exception:
                            pass

                        # Per-phoneme metrics (sparse — only on first few samples)
                        if batch_idx < 3 and b < 3 and self.phoneme_inventory:
                            try:
                                pp = compute_per_phoneme_metrics(
                                    pred_mel, gt_mel_sample, gt_phon,
                                    phoneme_classes=self.phoneme_inventory,
                                )
                                for ph, metrics in pp.items():
                                    if ph not in all_per_phoneme:
                                        all_per_phoneme[ph] = {
                                            "mcd": [], "rmse": [], "n_frames": [], "stoi": []}
                                    all_per_phoneme[ph]["mcd"].append(
                                        metrics["mcd"])
                                    all_per_phoneme[ph]["rmse"].append(
                                        metrics["rmse"])
                                    all_per_phoneme[ph]["n_frames"].append(
                                        metrics["n_frames"])
                            except Exception:
                                pass

                    # -- Group 2: Waveform metrics (vocode a subset) --
                    if compute_waveform_metrics and self.vocoder is not None and samples_vocoded < vocode_samples:
                        pred_wav = _vocode_to_waveform(
                            pred_mel, self.vocoder, self.hyperparameters, pred_len)
                        gt_wav = _vocode_to_waveform(
                            gt_mel_sample, self.vocoder, self.hyperparameters, gt_len)

                        if pred_wav is not None and gt_wav is not None:
                            try:
                                stoi_v = compute_stoi(pred_wav, gt_wav, 44100)
                            except Exception:
                                stoi_v = float("nan")
                            try:
                                pesq_v = compute_pesq(pred_wav, gt_wav, 44100)
                            except Exception:
                                pesq_v = float("nan")
                            try:
                                dnsmos_v = compute_dnsmos(pred_wav, 44100)
                            except Exception:
                                dnsmos_v = float("nan")
                            try:
                                wmse_v = compute_waveform_mse(pred_wav, gt_wav)
                            except Exception:
                                wmse_v = float("nan")
                            try:
                                f0_corr_v = compute_f0_correlation(
                                    pred_wav, gt_wav, 44100)
                            except Exception:
                                f0_corr_v = float("nan")
                            try:
                                vu_acc_v = compute_voiced_unvoiced_accuracy(
                                    pred_audio=pred_wav, gt_audio=gt_wav, sample_rate=44100,
                                )
                            except Exception:
                                vu_acc_v = float("nan")

                            accum["stoi"].append(stoi_v)
                            accum["pesq"].append(pesq_v)
                            accum["dnsmos"].append(dnsmos_v)
                            accum["waveform_mse"].append(wmse_v)
                            accum["f0_corr"].append(f0_corr_v)
                            accum["voiced_unvoiced_acc"].append(vu_acc_v)

                            # Duration correlation
                            if phon_len > 0:
                                try:
                                    pred_indices = phoneme_logits[b, :phon_len].argmax(
                                        dim=-1)
                                    dur_corr = compute_phoneme_duration_correlation(
                                        pred_indices, gt_phon,
                                    )
                                except Exception:
                                    dur_corr = float("nan")
                                accum["duration_corr"].append(dur_corr)

                            # WER / CER via ASR (uses a lazily-loaded whisper model)
                            gt_text = transcripts[b] if b < len(
                                transcripts) else ""
                            if gt_text:
                                asr = self._load_asr()
                                if asr is not None:
                                    try:
                                        wer_cer = compute_wer_cer(
                                            pred_wav, gt_text, 44100,
                                            device="cuda" if str(self.device).startswith(
                                                "cuda") else "cpu",
                                            asr_model=asr,
                                        )
                                        wer_value = wer_cer.get(
                                            "wer", float("nan"))
                                        cer_value = wer_cer.get(
                                            "cer", float("nan"))
                                        if np.isfinite(wer_value):
                                            accum["wer"].append(
                                                float(wer_value))
                                        if np.isfinite(cer_value):
                                            accum["cer"].append(
                                                float(cer_value))

                                        pp_stoi = compute_per_phoneme_stoi(
                                            pred_wav, gt_wav, gt_phon,
                                            sample_rate=44100,
                                            hop_length=getattr(
                                                self.hyperparameters, 'hop_length', 256),
                                            phoneme_classes=self.phoneme_inventory,
                                        )
                                        for ph, stoi_val in pp_stoi.items():
                                            all_per_phoneme.setdefault(
                                                ph, {"mcd": [], "rmse": [], "n_frames": [], "stoi": []})
                                            all_per_phoneme[ph].setdefault(
                                                "stoi", []).append(stoi_val)
                                    except Exception:
                                        pass

                            samples_vocoded += 1

        elapsed = time.perf_counter() - start_time

        # --- Aggregate results ---
        report = self._aggregate(accum, all_confusion,
                                 all_per_phoneme, elapsed)

        # --- Efficiency metrics (computed once, not per-sample) ---
        if compute_efficiency:
            report["efficiency"] = {
                "total_params": count_parameters(model)[0],
                "trainable_params": count_parameters(model)[1],
                "flops": compute_flops(model, device=self.device),
                "rtf": compute_rtf(model, dataloader, self.hyperparameters,
                                   device=self.device, train_on=train_on),
                "peak_memory_mb": measure_peak_memory(model, dataloader,
                                                      self.hyperparameters,
                                                      device=self.device,
                                                      train_on=train_on),
                "model_size_mb": measure_model_size(self.checkpoint_path) if self.checkpoint_path else float("nan"),
                "throughput_sps": measure_throughput(model, dataloader,
                                                     self.hyperparameters,
                                                     device=self.device,
                                                     train_on=train_on),
            }

        if training_history:
            report["training_dynamics"] = self._compute_training_dynamics(
                training_history)

        return report

    def _init_accumulators(self) -> Dict[str, list]:
        return {
            # Group 1
            "mcd": [], "lsd": [], "mel_rmse": [], "pearson_mean": [],
            # Group 2
            "stoi": [], "pesq": [], "dnsmos": [],
            "waveform_mse": [],
            # Group 3
            "per": [], "wer": [], "cer": [],
            # Group 4
            "dtw_cost": [], "f0_corr": [], "voiced_unvoiced_acc": [],
            "duration_corr": [],
        }

    def _load_asr(self):
        """Lazily load a whisper model once for WER/CER evaluation."""
        if getattr(self, "_asr_model", None) is None:
            self._asr_model = None
            try:
                if not hasattr(whisper, "load_model"):
                    raise ImportError(
                        "Installed 'whisper' is not openai-whisper; run: pip install openai-whisper"
                    )
                device = self.device if str(
                    self.device).startswith("cuda") else "cpu"
                self._asr_model = whisper.load_model("base", device=device)
            except Exception as exc:
                warnings.warn(f"Could not load whisper for WER/CER: {exc}")
        return self._asr_model

    @staticmethod
    def _compute_training_dynamics(history: Dict[str, Any]) -> Dict[str, Any]:
        """Derive training-dynamics metrics (group 6) from a training history."""
        val_loss = list(history.get("val_loss") or [])
        train_loss = list(history.get("train_loss") or [])
        if not val_loss:
            return {
                "convergence_epoch": float("nan"),
                "best_epoch": float("nan"),
                "best_val_loss": float("nan"),
                "generalization_gap": float("nan"),
                "epochs_run": 0,
            }

        best_idx = int(np.argmin(val_loss))
        best_val = float(val_loss[best_idx])
        threshold = best_val * 1.05 if best_val > 0 else best_val + 1e-6
        convergence = next((i for i, v in enumerate(
            val_loss) if v <= threshold), len(val_loss) - 1)
        gap = float(abs(train_loss[best_idx] - best_val)
                    ) if best_idx < len(train_loss) else float("nan")

        return {
            "convergence_epoch": int(convergence),
            "best_epoch": int(best_idx),
            "best_val_loss": best_val,
            "generalization_gap": gap,
            "epochs_run": len(val_loss),
        }

    @staticmethod
    def _safe_mean(values: list) -> float:
        """Mean of finite values, or NaN."""
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return float("nan")
        return float(np.mean(finite))

    @staticmethod
    def _safe_std(values: list) -> float:
        finite = [v for v in values if np.isfinite(v)]
        if len(finite) < 2:
            return float("nan")
        return float(np.std(finite))

    def _aggregate(
        self,
        accum: Dict[str, list],
        confusion_matrix: Optional[np.ndarray],
        per_phoneme: Dict[str, Dict[str, List[float]]],
        elapsed: float,
    ) -> Dict[str, Any]:
        """Aggregate accumulated per-sample metrics into summary statistics."""
        report: Dict[str, Any] = {}

        # Group 1: Spectral fidelity
        report["spectral"] = {
            "mcd_mean": self._safe_mean(accum["mcd"]),
            "mcd_std": self._safe_std(accum["mcd"]),
            "lsd_mean": self._safe_mean(accum["lsd"]),
            "lsd_std": self._safe_std(accum["lsd"]),
            "mel_rmse_mean": self._safe_mean(accum["mel_rmse"]),
            "mel_rmse_std": self._safe_std(accum["mel_rmse"]),
            "pearson_mean": self._safe_mean(accum["pearson_mean"]),
            "pearson_std": self._safe_std(accum["pearson_mean"]),
        }

        # Group 2: Perceptual quality
        report["perceptual"] = {
            "stoi_mean": self._safe_mean(accum["stoi"]),
            "stoi_std": self._safe_std(accum["stoi"]),
            "pesq_mean": self._safe_mean(accum["pesq"]),
            "pesq_std": self._safe_std(accum["pesq"]),
            "dnsmos_mean": self._safe_mean(accum["dnsmos"]),
            "dnsmos_std": self._safe_std(accum["dnsmos"]),
            "waveform_mse_mean": self._safe_mean(accum["waveform_mse"]),
            "waveform_mse_std": self._safe_std(accum["waveform_mse"]),
        }

        # Group 3: Intelligibility
        report["intelligibility"] = {
            "per_mean": self._safe_mean(accum["per"]),
            "per_std": self._safe_std(accum["per"]),
            "wer_mean": self._safe_mean(accum["wer"]),
            "wer_std": self._safe_std(accum["wer"]),
            "cer_mean": self._safe_mean(accum["cer"]),
            "cer_std": self._safe_std(accum["cer"]),
            "confusion_matrix": confusion_matrix,
            "confusion_labels": self.phoneme_inventory[:len(confusion_matrix)] if confusion_matrix is not None else [],
        }

        # Group 4: SSI-specific
        report["ssi_specific"] = {
            "dtw_cost_mean": self._safe_mean(accum["dtw_cost"]),
            "dtw_cost_std": self._safe_std(accum["dtw_cost"]),
            "f0_corr_mean": self._safe_mean(accum["f0_corr"]),
            "f0_corr_std": self._safe_std(accum["f0_corr"]),
            "voiced_unvoiced_acc_mean": self._safe_mean(accum["voiced_unvoiced_acc"]),
            "voiced_unvoiced_acc_std": self._safe_std(accum["voiced_unvoiced_acc"]),
            "duration_corr_mean": self._safe_mean(accum["duration_corr"]),
            "duration_corr_std": self._safe_std(accum["duration_corr"]),
            "per_phoneme": per_phoneme,
        }

        # Meta
        report["meta"] = {
            "n_samples_evaluated": len(accum["mcd"]),
            "n_samples_vocoded": len(accum["stoi"]),
            "elapsed_seconds": elapsed,
        }

        return report

    def log_to_mlflow(self, report: Dict[str, Any], step: int = 0) -> None:
        """Log all scalar metrics from the report to MLflow.

        Args:
            report: Evaluation report from self.evaluate().
            step: Step/epoch for MLflow logging.
        """
        try:
            import mlflow
        except ImportError:
            warnings.warn("mlflow not installed; skipping MLflow logging.")
            return

        # Group 1: Spectral
        for key, val in report.get("spectral", {}).items():
            if isinstance(val, (int, float)) and np.isfinite(val):
                mlflow.log_metric(
                    f"metrics/spectral/{key}", float(val), step=step)

        # Group 2: Perceptual
        for key, val in report.get("perceptual", {}).items():
            if isinstance(val, (int, float)) and np.isfinite(val):
                mlflow.log_metric(
                    f"metrics/perceptual/{key}", float(val), step=step)

        # Group 3: Intelligibility
        for key, val in report.get("intelligibility", {}).items():
            if isinstance(val, (int, float)) and np.isfinite(val):
                mlflow.log_metric(
                    f"metrics/intelligibility/{key}", float(val), step=step)

        # Group 4: SSI-specific
        for key, val in report.get("ssi_specific", {}).items():
            if key == "per_phoneme" or not isinstance(val, (int, float)):
                continue
            if np.isfinite(val):
                mlflow.log_metric(
                    f"metrics/ssi_specific/{key}", float(val), step=step)

        # Group 5: Efficiency
        for key, val in report.get("efficiency", {}).items():
            if isinstance(val, (int, float)) and np.isfinite(val):
                mlflow.log_metric(
                    f"metrics/efficiency/{key}", float(val), step=step)

        # Group 6: Training dynamics
        for key, val in report.get("training_dynamics", {}).items():
            if isinstance(val, (int, float)) and np.isfinite(val):
                mlflow.log_metric(
                    f"metrics/training_dynamics/{key}", float(val), step=step)

        # Meta
        for key, val in report.get("meta", {}).items():
            if isinstance(val, (int, float)) and np.isfinite(val):
                mlflow.log_metric(f"metrics/meta/{key}", float(val), step=step)

    @staticmethod
    def summary(report: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the evaluation report."""
        lines = ["=" * 60]
        lines.append("SSI Model Evaluation Summary")
        lines.append("=" * 60)

        meta = report.get("meta", {})
        lines.append(
            f"Samples evaluated: {meta.get('n_samples_evaluated', 'N/A')}")
        lines.append(
            f"Samples vocoded:   {meta.get('n_samples_vocoded', 'N/A')}")
        lines.append(
            f"Time elapsed:      {meta.get('elapsed_seconds', 0):.1f}s")
        lines.append("")

        # Group 1
        lines.append("--- Group 1: Spectral Reconstruction Fidelity ---")
        s = report.get("spectral", {})
        lines.append(
            f"  MCD (dB):         {s.get('mcd_mean', 'N/A'):.2f} ± {s.get('mcd_std', 'N/A'):.2f}")
        lines.append(
            f"  LSD (dB):         {s.get('lsd_mean', 'N/A'):.2f} ± {s.get('lsd_std', 'N/A'):.2f}")
        lines.append(
            f"  Mel RMSE:         {s.get('mel_rmse_mean', 'N/A'):.4f} ± {s.get('mel_rmse_std', 'N/A'):.4f}")
        lines.append(
            f"  Pearson r (mean): {s.get('pearson_mean', 'N/A'):.3f} ± {s.get('pearson_std', 'N/A'):.3f}")

        # Group 2
        lines.append("--- Group 2: Perceptual Audio Quality ---")
        p = report.get("perceptual", {})
        lines.append(
            f"  STOI:             {p.get('stoi_mean', 'N/A'):.3f} ± {p.get('stoi_std', 'N/A'):.3f}")
        lines.append(
            f"  PESQ:             {p.get('pesq_mean', 'N/A'):.2f} ± {p.get('pesq_std', 'N/A'):.2f}")
        lines.append(
            f"  DNSMOS:           {p.get('dnsmos_mean', 'N/A'):.2f} ± {p.get('dnsmos_std', 'N/A'):.2f}")
        lines.append(
            f"  Waveform MSE:     {p.get('waveform_mse_mean', 'N/A'):.6f}")

        # Group 3
        lines.append("--- Group 3: Intelligibility ---")
        i = report.get("intelligibility", {})
        lines.append(
            f"  PER:              {i.get('per_mean', 'N/A'):.3f} ± {i.get('per_std', 'N/A'):.3f}")
        lines.append(
            f"  WER:              {i.get('wer_mean', 'N/A'):.3f} ± {i.get('wer_std', 'N/A'):.3f}")
        lines.append(
            f"  CER:              {i.get('cer_mean', 'N/A'):.3f} ± {i.get('cer_std', 'N/A'):.3f}")

        # Group 4
        lines.append("--- Group 4: SSI-Specific Mapping ---")
        ss = report.get("ssi_specific", {})
        lines.append(
            f"  DTW Cost:         {ss.get('dtw_cost_mean', 'N/A'):.2f} ± {ss.get('dtw_cost_std', 'N/A'):.2f}")
        lines.append(
            f"  F0 Correlation:   {ss.get('f0_corr_mean', 'N/A'):.3f} ± {ss.get('f0_corr_std', 'N/A'):.3f}")
        lines.append(
            f"  V/UV Accuracy:    {ss.get('voiced_unvoiced_acc_mean', 'N/A'):.3f}")
        lines.append(
            f"  Duration Corr:    {ss.get('duration_corr_mean', 'N/A'):.3f}")

        # Group 5
        lines.append("--- Group 5: Computational Efficiency ---")
        e = report.get("efficiency", {})
        if e:
            lines.append(
                f"  Total Params:     {e.get('total_params', 'N/A'):,}")
            lines.append(f"  FLOPs:            {e.get('flops', 'N/A')}")
            lines.append(f"  RTF:              {e.get('rtf', 'N/A'):.4f}")
            lines.append(
                f"  Peak GPU Mem:     {e.get('peak_memory_mb', 'N/A'):.1f} MB")
            lines.append(
                f"  Model Size:       {e.get('model_size_mb', 'N/A'):.1f} MB")

        # Group 6
        td = report.get("training_dynamics", {})
        if td:
            lines.append("--- Group 6: Training Dynamics ---")
            lines.append(f"  Epochs Run:       {td.get('epochs_run', 'N/A')}")
            lines.append(f"  Best Epoch:       {td.get('best_epoch', 'N/A')}")
            lines.append(
                f"  Best Val Loss:    {td.get('best_val_loss', 'N/A'):.4f}")
            lines.append(
                f"  Convergence Ep.:  {td.get('convergence_epoch', 'N/A')}")
            lines.append(
                f"  General. Gap:     {td.get('generalization_gap', 'N/A'):.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

def evaluate_model(
    model: Module,
    dataloader: DataLoader,
    hyperparameters,
    phoneme_inventory: Optional[List[str]] = None,
    checkpoint_path: Optional[str] = None,
    max_batches: Optional[int] = None,
    log_to_mlflow: bool = True,
    step: int = 0,
    training_history: Optional[Dict[str, Any]] = None,
    train_on: str = "voiced",
) -> Dict[str, Any]:
    """One-liner: evaluate a model and optionally log to MLflow."""
    suite = MetricsSuite(
        model=model,
        hyperparameters=hyperparameters,
        phoneme_inventory=phoneme_inventory,
        checkpoint_path=checkpoint_path,
    )

    report = suite.evaluate(
        dataloader,
        max_batches=max_batches,
        train_on=train_on,
        training_history=training_history,
    )

    if log_to_mlflow:
        suite.log_to_mlflow(report, step=step)

    return report