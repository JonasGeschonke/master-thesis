import torch
from datetime import datetime

class Hyperparameters:
    grad_clip = 1.0
    mixed_precision = True
    seed = 3401
    sample_rate = 44100
    learning_rate = 1e-3
    hop_length = 256
    batch_size = 16
    save_dir = './models/checkpoints'
    weight_decay = 1e-4
    experiment_name = 'transformer_experiment'
    device = 'cuda'
    freq_cutoff = 17000
    n_fft = 1024
    n_mels = 128
    win_length = 1024
    num_workers = 4
    epochs = 600
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    phoneme_loss_weight_init = 0.5
    normalize_loss_by_sample = True
    # Warmup epochs for learning rate (linear warmup)
    warmup_epochs = 5

    # --- 'both' input-branch weighting / curriculum (V/U/B study) ---
    both_voiced_weight = 1.0            # multiplier on the voiced-view loss
    both_unvoiced_weight = 1.0          # multiplier on the unvoiced-view loss
    unvoiced_curriculum_epochs = 0      # ramp unvoiced 0 -> both_unvoiced_weight
                                        # over the first N epochs (0 = no ramp)
    early_stop_source = "both"          # 'both' | 'voiced' | 'unvoiced':
                                        # which source's val loss drives early stopping

    # --- Multi-Scale Mel Loss (MSSC) — Phase 0 spectral detail loss ---
    mssc_enabled = True
    mssc_loss_weight = 0.75          # initial weight (ramps up over warmup)
    mssc_loss_weight_warmup = 10     # epochs to ramp from 0 → full
    mssc_scales = [0.0, 1.0, 2.0, 4.0]  # Gaussian blur σ in mel bins
    mssc_sc_weight = 1.0             # spectral convergence multiplier
    mssc_mag_weight = 1.0            # log-magnitude L1 multiplier

    def __init__(self):
        self._set_device()
        # add timestamp to save_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"{self.save_dir}/{timestamp}"

    def _set_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda:5'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

    def set_training_params(self, learning_rate: float, weight_decay: float):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def __dict__(self):
        return {
            "grad_clip": self.grad_clip,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "sample_rate": self.sample_rate,
            "learning_rate": self.learning_rate,
            "hop_length": self.hop_length,
            "batch_size": self.batch_size,
            "save_dir": self.save_dir,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "freq_cutoff": self.freq_cutoff,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "epochs": self.epochs,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "phoneme_loss_weight_init": self.phoneme_loss_weight_init,
            "normalize_loss_by_sample": self.normalize_loss_by_sample,
            "warmup_epochs": self.warmup_epochs,
            # MSSC Phase 0
            "mssc_enabled": self.mssc_enabled,
            "mssc_loss_weight": self.mssc_loss_weight,
            "mssc_loss_weight_warmup": self.mssc_loss_weight_warmup,
            "mssc_scales": str(self.mssc_scales),
            "mssc_sc_weight": self.mssc_sc_weight,
            "mssc_mag_weight": self.mssc_mag_weight,
        }