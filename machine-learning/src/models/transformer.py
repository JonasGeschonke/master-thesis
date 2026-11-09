import sys
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch.nn import Dropout, Module, Linear, GELU, Sequential, TransformerEncoder, TransformerEncoderLayer, init, BatchNorm1d, Conv1d, ModuleList, functional as F
from torch import Tensor, no_grad
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

_src_root = Path(__file__).resolve().parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from criterion.dtw_loss import dtw_loss_with_phonemes_and_alignment
from utils.utils_transformer import PHONEME_INVENTORY
from criterion.multi_scale_mel_loss import MultiScaleMelLoss

def _prepare_aligned_batch(output_seq, ground_truth_seq, output_lengths, alignments):
    batch_size = output_seq.size(0)
    n_feats = output_seq.size(-1)
    valid_lens = [int(output_lengths[i].item()) for i in range(batch_size)]
    max_len = max(valid_lens) if valid_lens else 0

    pred_batch = output_seq[:batch_size, :max_len, :].contiguous()
    gt_aligned = ground_truth_seq.new_zeros(batch_size, max_len, n_feats)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=output_seq.device)

    for i in range(batch_size):
        v = valid_lens[i]
        if v == 0:
            continue
        gt_aligned[i, :v] = ground_truth_seq[i, alignments[i], :]
        mask[i, :v] = True

    return pred_batch, gt_aligned, mask

def init_weights(module):
    if isinstance(module, Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

    elif isinstance(module, TransformerEncoderLayer):
        for param in module.parameters():
            if param.dim() > 1:  # Weights
                init.xavier_uniform_(param)
            else:  # Biases
                init.zeros_(param)


def visualize_tensor(tensor: torch.Tensor, step_name: str):
    with torch.no_grad():
        if tensor.dim() == 3:
            t_2d = tensor[0].cpu().numpy()
        elif tensor.dim() == 2:
            t_2d = tensor.cpu().numpy()
        else:
            print(
                f"[{step_name}] Cannot plot tensor with {tensor.dim()} dimensions.")
            return

        print(f"[{step_name}] Visualizing tensor with shape: {tensor.shape}")
        t_plot = t_2d.T

        # 3. Plotting
        plt.figure(figsize=(10, 4))
        plt.imshow(t_plot, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Forward Pass Layer: {step_name}\nShape: {tensor.shape}")
        plt.xlabel("Time Step (Sequence Length)")
        plt.ylabel("Feature / Embedding Dimension")
        plt.colorbar(label="Activation Value")

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig(f"{step_name}.png")

class ResBlock1D(Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(channels)
        self.conv2 = Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # type: ignore
        return x

class PostNet(Module):
    """5-Layer 1D CNN for Mel-Spectrogram projection."""
    def __init__(self, n_mels: int = 128, d_model: int = 512, kernel_size: int = 5, dropout: float = 0.3):
        super().__init__()
        self.convs = ModuleList([
            Sequential(
                Conv1d(n_mels if i == 0 else d_model,
                    d_model if i < 4 else n_mels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                BatchNorm1d(d_model if i < 4 else n_mels),
                GELU(),
                Dropout(dropout)
            )
            for i in range(5)
        ])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class TransformerModel(Module):
    def __init__(self, input_dim_lp: int = 128, model_dim: int = 512, doppler_dim: int = 1024, num_heads: int = 4, num_layers: int = 4, phoneme_inventory: list[str] | None = None, dropout: float = 0.1, loss_weight_init: float = 0.5, n_mels: int = 128):
        super().__init__()
        # --- Input Streams (Phase Gradient, Amplitude Gradient, Doppler Spectrum, Residual Speech Features) ---
        self.pha_embed = Sequential(
            Conv1d(1, 64, kernel_size=5, padding=2), ResBlock1D(64))
        self.amp_embed = Sequential(
            Conv1d(1, 64, kernel_size=5, padding=2), ResBlock1D(64))
        self.dop_embed = Sequential(
            Conv1d(doppler_dim, 64, kernel_size=5, padding=2), ResBlock1D(64))
        self.lp_embed = Sequential(
            BatchNorm1d(input_dim_lp),
            Conv1d(input_dim_lp, 64, kernel_size=5, padding=2), ResBlock1D(64))

        self.dilated_convs = ModuleList([
            Conv1d(64 * 4 if i == 0 else model_dim, model_dim,
                   kernel_size=3, padding=2**i, dilation=2**i)
            for i in range(6)
        ])

        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # --- Mel Decoder ---
        self.mel_coarse = Linear(model_dim, n_mels)
        self.mel_postnet = PostNet(n_mels=n_mels, d_model=model_dim)

        # --- Phoneme Prediction Head ---
        self.phoneme_inventory = phoneme_inventory or PHONEME_INVENTORY
        self.phoneme_prediction = Linear(
            model_dim, len(self.phoneme_inventory))

    def _initialize_gate(self):
        gate_linear: Linear = self.gate[0]
        init.normal_(gate_linear.weight, mean=0.0, std=0.01)
        init.constant_(gate_linear.bias, -3.0)

    def forward(self, padded_sequence_lp: Tensor, padded_sequence_pha: Tensor, padded_sequence_amp: Tensor, padded_sequence_dop: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if padded_sequence_pha.dim() == 2:
            padded_sequence_pha = padded_sequence_pha.unsqueeze(-1)
        if padded_sequence_amp.dim() == 2:
            padded_sequence_amp = padded_sequence_amp.unsqueeze(-1)

        mask = torch.arange(padded_sequence_pha.size(
            1), device=padded_sequence_pha.device).unsqueeze(0) >= lengths.unsqueeze(1)
        # Pass Features through their respective embedding streams
        pha_feat = self.pha_embed(
            padded_sequence_pha.permute(0, 2, 1))
        amp_feat = self.amp_embed(
            padded_sequence_amp.permute(0, 2, 1))
        dop_feat = self.dop_embed(
            padded_sequence_dop.permute(0, 2, 1))
        lp_feat = self.lp_embed(
            padded_sequence_lp.permute(0, 2, 1)
        )

        # Concatenate along channel dimension
        fused_features = torch.cat(
            [pha_feat, amp_feat, dop_feat, lp_feat], dim=1)

        # Apply dilated convolutions
        for d_conv in self.dilated_convs:
            fused_features = F.relu(
                d_conv(fused_features))

        # Main encoder stream: UltraSR features → transformer
        fused_features = fused_features.permute(0, 2, 1)
        x_main = self.positional_encoding(fused_features)
        h_main = self.transformer_encoder(x_main, src_key_padding_mask=mask)

        h_fused = h_main
        h_fused = h_fused.masked_fill(mask.unsqueeze(-1), 0.0)

        # Decode to mel-spectrogram and phonemes
        mel_coarse = self.mel_coarse(h_fused)
        mel_residual = self.mel_postnet(mel_coarse.permute(0, 2, 1))
        mel_residual = mel_residual.permute(0, 2, 1)
        mel_out = mel_coarse + mel_residual
        phoneme_out = self.phoneme_prediction(h_fused)
        return mel_out, lengths, phoneme_out

    def _both_source_weight(self, src: str, epoch: int, hyperparameters, train_on: str,
                            apply_curriculum: bool = True) -> float:
        """Per-source loss weight for the 'both' input-branch training mode."""
        if train_on != "both":
            return 1.0
        wv = float(getattr(hyperparameters, "both_voiced_weight", 1.0))
        wu = float(getattr(hyperparameters, "both_unvoiced_weight", 1.0))
        if wv <= 0.0 and wu <= 0.0:
            wv = wu = 1.0
        ramp = int(getattr(hyperparameters, "unvoiced_curriculum_epochs", 0) or 0)
        if apply_curriculum and ramp > 0:
            alpha = min(1.0, float(epoch + 1) / float(ramp))
            wv = (1.0 - alpha) * 1.0 + alpha * wv
            wu = (1.0 - alpha) * 1.0 + alpha * wu
        if wu > wv and wv > 0.0:
            # Emphasise the unvoiced view without reducing the voiced weight.
            wv_eff, wu_eff = wv, wu
        else:
            total = wv + wu
            if total > 0.0:
                scale = 2.0 / total
                wv_eff, wu_eff = wv * scale, wu * scale
            else:
                wv_eff = wu_eff = 1.0
        return wv_eff if src == "voiced" else wu_eff

    def train_epoch(
        self,
        optimizer: Optimizer,
        train_loader: DataLoader,
        hyperparameters,
        mssc_loss_fn: MultiScaleMelLoss,
        epoch: int = 0,
        train_on: str = 'unvoiced',
    ) -> tuple:
        # --- Set model to training mode and reset epoch-level loss accumulators ---
        self.train()
        total_epoch_loss = 0.0
        total_mel_loss = 0.0
        total_phon_loss = 0.0
        total_phoneme_acc = 0.0
        total_mssc_sc = 0.0
        total_mssc_mag = 0.0
        batch_count = 0

        if not hasattr(hyperparameters, 'loss_ema'):
            hyperparameters.loss_ema = {
                'mel': None,
                'phon': None, 
                'mom': 0.995, 
                'eps': 1e-8
            }

        # --- Automatic Mixed Precision (AMP) ---
        if not hasattr(self, '_scaler'):
            self._scaler = torch.amp.GradScaler() if isinstance(hyperparameters.device, str) and hyperparameters.device.startswith('cuda') else None

        # Per-source unweighted metrics (voiced vs unvoiced) for diagnostics.
        src_totals = {}

        for batch in train_loader:
            # --- Determine which data sources to process ---
            sources = ['voiced', 'unvoiced'] if train_on == 'both' else [train_on]

            optimizer.zero_grad()

            for src in sources:
                # --- Extract training data based on the current source ---
                low_pass_sequence = batch[src]['lp']
                pha_sequence = batch[src]['pha']
                amp_sequence = batch[src]['amp']
                dop_sequence = batch[src]['dop']
                lengths = batch[src]['lengths']

                # --- Move data to device ---
                low_pass_sequence = low_pass_sequence.to(hyperparameters.device)
                pha_sequence = pha_sequence.to(hyperparameters.device)
                amp_sequence = amp_sequence.to(hyperparameters.device)
                dop_sequence = dop_sequence.to(hyperparameters.device)
                lengths = lengths.to(hyperparameters.device)

                # --- Forward pass through the model (AMP) ---
                with torch.autocast(device_type='cuda', enabled=self._scaler is not None):
                    output_seq, output_lengths, phonemes = self.forward(
                        low_pass_sequence,
                        pha_sequence,
                        amp_sequence,
                        dop_sequence,
                        lengths=lengths,
                    )
                output_seq = output_seq.float()
                phonemes = phonemes.float()

                # --- Load ground truth phonemes and mel spectrograms (same for both sources) ---
                phonemes_gt = batch['phoneme']['data'].to(hyperparameters.device)
                phoneme_lengths = batch['phoneme']['lengths'].to(
                    hyperparameters.device)
                ground_truth_seq = batch['gt']['data'].to(hyperparameters.device)
                lengths_ground_truth = batch['gt']['lengths'].to(
                    hyperparameters.device)

                # --- Compute losses ---
                mel_loss, phon_loss, phoneme_acc, alignments = dtw_loss_with_phonemes_and_alignment(
                    output_seq, ground_truth_seq, output_lengths, lengths_ground_truth,
                    phonemes, phonemes_gt, phoneme_lengths, phoneme_lengths,
                    normalize_by_sample=hyperparameters.normalize_loss_by_sample,
                    use_delta_cost=hyperparameters.dtw_use_delta_cost,
                    use_band_weight=hyperparameters.dtw_use_band_weight,
                    use_l1_cost=hyperparameters.dtw_use_l1_cost,
                    use_contrast_loss=hyperparameters.dtw_use_contrast_loss,
                    contrast_weight=hyperparameters.dtw_contrast_weight,
                    use_post_alignment_l1=hyperparameters.dtw_use_post_alignment_l1,
                    post_alignment_weight=hyperparameters.dtw_post_alignment_weight,
                    band_decay=hyperparameters.dtw_band_decay,
                )

                # --- Normalize losses by their running magnitudes (EMA) ---
                with torch.no_grad():
                    ema = hyperparameters.loss_ema
                    # Safely extract scalar magnitudes
                    try:
                        mel_val = mel_loss.detach().cpu().item()
                    except Exception:
                        mel_val = float(mel_loss)
                    try:
                        phon_val = phon_loss.detach().cpu().item()
                    except Exception:
                        phon_val = float(phon_loss)

                    if ema['mel'] is None:
                        ema['mel'] = mel_val
                        ema['phon'] = phon_val
                    else:
                        mom = ema.get('mom', 0.995)
                        ema['mel'] = mom * ema['mel'] + (1.0 - mom) * mel_val
                        ema['phon'] = mom * ema['phon'] + (1.0 - mom) * phon_val

                # --- Normalize losses (detach EMA to avoid gradients flowing through it) ---
                norm_mel = mel_loss / \
                    (hyperparameters.loss_ema['mel'] +
                     hyperparameters.loss_ema['eps'])
                norm_phon = phon_loss / \
                    (hyperparameters.loss_ema['phon'] +
                     hyperparameters.loss_ema['eps'])

                loss = norm_mel + (hyperparameters.phoneme_loss_weight_init * norm_phon)

                # --- Multi-Scale Mel Loss (MSSC) on DTW-aligned sequences ---
                pred_batch, gt_aligned, loss_mask = _prepare_aligned_batch(
                    output_seq, ground_truth_seq, output_lengths, alignments
                )
                mssc_sc, mssc_mag = mssc_loss_fn(pred_batch, gt_aligned, mask=loss_mask)

                warmup_epochs = getattr(
                    hyperparameters, 'mssc_loss_weight_warmup', 10)
                warmup_factor = min(1.0, float(epoch) /
                                    float(max(1, warmup_epochs)))
                mssc_weight = warmup_factor * hyperparameters.mssc_loss_weight

                sc_w = getattr(hyperparameters, 'mssc_sc_weight', 1.0)
                mag_w = getattr(hyperparameters, 'mssc_mag_weight', 1.0)
                loss = loss + (mssc_weight * (sc_w * mssc_sc + mag_w * mssc_mag))

                _src_w = self._both_source_weight(src, epoch, hyperparameters, train_on)
                if _src_w != 1.0:
                    loss = loss * _src_w

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

                _st = src_totals.setdefault(src, {"mel": 0.0, "phon": 0.0, "acc": 0.0, "n": 0})
                _st["mel"] += mel_loss.item() if isinstance(mel_loss, torch.Tensor) else float(mel_loss)
                _st["phon"] += phon_loss.item() if isinstance(phon_loss, torch.Tensor) else float(phon_loss)
                _st["acc"] += float(phoneme_acc.detach().cpu()) if isinstance(phoneme_acc, torch.Tensor) else float(phoneme_acc)
                _st["n"] += 1

                total_epoch_loss += loss.item()
                total_mel_loss += mel_loss.item() if isinstance(mel_loss,
                                                                torch.Tensor) else mel_loss
                total_phon_loss += phon_loss.item() if isinstance(phon_loss,
                                                                  torch.Tensor) else phon_loss
                total_phoneme_acc += phoneme_acc if isinstance(
                    phoneme_acc, float) else phoneme_acc
                total_mssc_sc += mssc_sc.item() if isinstance(mssc_sc, torch.Tensor) else mssc_sc
                total_mssc_mag += mssc_mag.item() if isinstance(mssc_mag,
                                                                torch.Tensor) else mssc_mag
                batch_count += 1

            # --- Gradient clipping (configurable via hyperparameters.grad_clip) ---
            try:
                max_norm = float(getattr(hyperparameters, 'grad_clip', 1.0))
            except Exception:
                max_norm = 1.0
            if self._scaler is not None:
                self._scaler.unscale_(optimizer)
            clip_grad_norm_(self.parameters(), max_norm=max_norm)
            if self._scaler is not None:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()

        hyperparameters.last_train_source_metrics = {
            k: {"mel_loss": v["mel"] / max(1, v["n"]),
                "phon_loss": v["phon"] / max(1, v["n"]),
                "phoneme_acc": v["acc"] / max(1, v["n"])}
            for k, v in src_totals.items()
        }

        # --- Compute epoch averages ---
        average_epoch_loss = total_epoch_loss / batch_count if batch_count > 0 else 0.0
        average_mel_loss = total_mel_loss / batch_count if batch_count > 0 else 0.0
        average_phon_loss = total_phon_loss / batch_count if batch_count > 0 else 0.0
        average_phoneme_acc = total_phoneme_acc / \
            batch_count if batch_count > 0 else 0.0
        average_mssc_sc = total_mssc_sc / batch_count if batch_count > 0 else 0.0
        average_mssc_mag = total_mssc_mag / batch_count if batch_count > 0 else 0.0
        std_epoch_loss = 0.0
        return (average_epoch_loss, average_phoneme_acc, average_mel_loss, average_phon_loss,
                average_mssc_sc, average_mssc_mag, std_epoch_loss)

    def evaluate(
        self,
        dataloader: DataLoader,
        hyperparameters,
        mssc_loss_fn: MultiScaleMelLoss,
        epoch: int,
        train_on: str = 'unvoiced',
    ):
        self.eval()
        val_loss = 0.0
        val_batch_count = 0
        val_phon_loss_total = 0.0
        val_mel_loss_total = 0.0
        val_phon_acc_total = 0.0
        val_mssc_sc_total = 0.0
        val_mssc_mag_total = 0.0

        src_totals = {}

        with no_grad():
            for batch in dataloader:
                # --- Determine which data sources to process ---
                sources = ['voiced', 'unvoiced'] if train_on == 'both' else [train_on]

                for src in sources:
                    low_pass_sequence = batch[src]['lp']
                    pha_sequence = batch[src]['pha']
                    amp_sequence = batch[src]['amp']
                    dop_sequence = batch[src]['dop']
                    lengths = batch[src]['lengths']

                    low_pass_sequence = low_pass_sequence.to(
                        hyperparameters.device)
                    pha_sequence = pha_sequence.to(hyperparameters.device)
                    amp_sequence = amp_sequence.to(hyperparameters.device)
                    dop_sequence = dop_sequence.to(hyperparameters.device)
                    lengths = lengths.to(hyperparameters.device)

                    with torch.autocast(device_type='cuda', enabled=isinstance(hyperparameters.device, str) and hyperparameters.device.startswith('cuda')):
                        outputs_seq, output_lengths, phonemes = self.forward(
                            low_pass_sequence,
                            pha_sequence,
                            amp_sequence,
                            dop_sequence,
                            lengths=lengths,
                        )
                    outputs_seq = outputs_seq.float()
                    phonemes = phonemes.float()

                    phonemes_gt = batch['phoneme']['data'].to(
                        hyperparameters.device)
                    phoneme_lengths = batch['phoneme']['lengths'].to(
                        hyperparameters.device)
                    ground_truth_seq = batch['gt']['data'].to(
                        hyperparameters.device)
                    lengths_ground_truth = batch['gt']['lengths'].to(
                        hyperparameters.device)

                    mel_loss, phon_loss, phoneme_acc, alignments = dtw_loss_with_phonemes_and_alignment(
                        outputs_seq, ground_truth_seq, output_lengths, lengths_ground_truth,
                        phonemes, phonemes_gt, phoneme_lengths, phoneme_lengths,
                        normalize_by_sample=hyperparameters.normalize_loss_by_sample,
                        use_delta_cost=hyperparameters.dtw_use_delta_cost,
                        use_band_weight=hyperparameters.dtw_use_band_weight,
                        use_l1_cost=hyperparameters.dtw_use_l1_cost,
                        use_contrast_loss=hyperparameters.dtw_use_contrast_loss,
                        contrast_weight=hyperparameters.dtw_contrast_weight,
                        use_post_alignment_l1=hyperparameters.dtw_use_post_alignment_l1,
                        post_alignment_weight=hyperparameters.dtw_post_alignment_weight,
                        band_decay=hyperparameters.dtw_band_decay,
                    )

                    total_val_loss = mel_loss + (hyperparameters.phoneme_loss_weight_init * phon_loss)

                    pred_batch, gt_aligned, loss_mask = _prepare_aligned_batch(
                        outputs_seq, ground_truth_seq, output_lengths, alignments
                    )
                    mssc_sc, mssc_mag = mssc_loss_fn(pred_batch, gt_aligned, mask=loss_mask)
                    val_mssc_sc_total += mssc_sc.item()
                    val_mssc_mag_total += mssc_mag.item()

                    _src_w = self._both_source_weight(src, epoch, hyperparameters, train_on,
                                                      apply_curriculum=False)

                    val_loss += (_src_w * total_val_loss.item())
                    val_mel_loss_total += mel_loss.item()
                    val_phon_loss_total += phon_loss.item()
                    val_phon_acc_total += phoneme_acc
                    val_batch_count += 1

                    _st = src_totals.setdefault(src, {"mel": 0.0, "phon": 0.0, "acc": 0.0, "n": 0})
                    _st["mel"] += mel_loss.item()
                    _st["phon"] += phon_loss.item()
                    _st["acc"] += float(phoneme_acc.detach().cpu()) if isinstance(phoneme_acc, torch.Tensor) else float(phoneme_acc)
                    _st["n"] += 1

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
        avg_val_mel_loss = val_mel_loss_total / \
            val_batch_count if val_batch_count > 0 else 0.0
        avg_val_phon_loss = val_phon_loss_total / \
            val_batch_count if val_batch_count > 0 else 0.0
        avg_val_phon_acc = val_phon_acc_total / \
            val_batch_count if val_batch_count > 0 else 0.0
        avg_val_mssc_sc = val_mssc_sc_total / \
            val_batch_count if val_batch_count > 0 else 0.0
        avg_val_mssc_mag = val_mssc_mag_total / \
            val_batch_count if val_batch_count > 0 else 0.0
        hyperparameters.last_val_source_metrics = {
            k: {"mel_loss": v["mel"] / max(1, v["n"]),
                "phon_loss": v["phon"] / max(1, v["n"]),
                "phoneme_acc": v["acc"] / max(1, v["n"])}
            for k, v in src_totals.items()
        }
        return (avg_val_loss, avg_val_phon_acc, avg_val_mel_loss, avg_val_phon_loss,
                avg_val_mssc_sc, avg_val_mssc_mag, 0.0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def plot(
        self,
        padded_sequence_lp: Tensor,
        padded_sequence_pha: Tensor,
        padded_sequence_amp: Tensor,
        padded_sequence_dop: Tensor,
        lengths: Tensor
    ):
        """
        Forward pass with visualization of intermediate tensors.
        """
        pha_feat = self.pha_embed(
            padded_sequence_pha.permute(0, 2, 1))  # [B, 64, T]
        visualize_tensor(pha_feat, "Phase Embedding")

        amp_feat = self.amp_embed(
            padded_sequence_amp.permute(0, 2, 1))  # [B, 64, T]
        visualize_tensor(amp_feat, "Amplitude Embedding")

        dop_feat = self.dop_embed(
            padded_sequence_dop.permute(0, 2, 1))  # [B, 64, T]
        visualize_tensor(dop_feat, "Doppler Embedding")

        outputs_seq, output_lengths, phonemes = self.forward(
            padded_sequence_lp, pha_feat, amp_feat, dop_feat, lengths=lengths)
        visualize_tensor(outputs_seq, "Transformer Output")

if __name__ == "__main__":
    model = TransformerModel(input_dim_lp=128,
                             model_dim=512, num_heads=4, num_layers=4, dropout=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size, time_steps = 4, 200
    dummy_lp = torch.randn(batch_size, time_steps, 128)
    dummy_pha = torch.randn(batch_size, time_steps, 1)
    dummy_amp = torch.randn(batch_size, time_steps, 1)
    dummy_dop = torch.randn(batch_size, time_steps, 1024)
    dummy_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)

    mel_out, out_lengths, phoneme_out = model(
        dummy_lp, dummy_pha, dummy_amp, dummy_dop, dummy_lengths)
    print(f"\nForward pass with four input streams and post-net decoder:")
    print(f"  Mel output:    {tuple(mel_out.shape)}")
    print(f"  Phoneme out:   {tuple(phoneme_out.shape)}")
    print(model)
