import random
import torch
import numpy as np
import soundfile as sf

from torch import Tensor, tensor
from torch.utils.data import Dataset
from pathlib import Path
from torchaudio.transforms import Spectrogram
from torch.nn.utils.rnn import pad_sequence

from dataset.textgrid import TextGrid
from dataset.residual_features import ResidualFeatureExtractor
from utils.ultrasr_features import UltraSRFeatureExtractor, load_ultrasr_features
from utils.hyperparameters_transformer import Hyperparameters
from utils.utils_transformer import (
    _resample_to_target_sr,
    load_audio,
    textgrid_to_phoneme_sequence,
)

def textgrid_to_transcript(textgrid_path: Path) -> str:
    """Extract the utterance text from the 'words' tier of a TextGrid."""
    try:
        textgrid = TextGrid.load(textgrid_path)
    except Exception:
        return ""
    words_tier = None
    for tier in textgrid.tiers:
        if getattr(tier, 'nameid', '') == 'words':
            words_tier = tier
            break
    if words_tier is None and textgrid.tiers:
        words_tier = textgrid.tiers[0]
    if words_tier is None:
        return ""
    words = [str(label).strip() for (_, _, label) in words_tier.simple_transcript if str(label).strip()]
    return " ".join(words)

def pad_collate_fn(batch: list[dict]) -> dict[str, dict[str, Tensor]]:
    spec_train_lp = [item['training_data'][0][0] for item in batch]
    spec_train_pha = [item['training_data'][0][1] for item in batch]
    spec_train_amp = [item['training_data'][0][2] for item in batch]
    spec_train_dop = [item['training_data'][0][3] for item in batch]
    spec_gt_lp = [item['training_data'][1][0] for item in batch]
    spec_gt_pha = [item['training_data'][1][1] for item in batch]
    spec_gt_amp = [item['training_data'][1][2] for item in batch]
    spec_gt_dop = [item['training_data'][1][3] for item in batch]
    phonemes: list[Tensor] = [item['phoneme'] for item in batch]
    gt: list[Tensor] = [item['gt'] for item in batch]

    len_spec_train = tensor([t.shape[0] for t in spec_train_lp])
    len_spec_gt = tensor([t.shape[0] for t in spec_gt_lp])
    len_gt = tensor([t.shape[0] for t in gt])
    len_phonemes = tensor([t.shape[0] for t in phonemes])

    spec_train_lp_padded = pad_sequence(spec_train_lp, batch_first=True)
    spec_train_pha_padded = pad_sequence(spec_train_pha, batch_first=True)
    spec_train_amp_padded = pad_sequence(spec_train_amp, batch_first=True)
    spec_train_dop_padded = pad_sequence(spec_train_dop, batch_first=True)
    spec_gt_lp_padded = pad_sequence(spec_gt_lp, batch_first=True)
    spec_gt_pha_padded = pad_sequence(spec_gt_pha, batch_first=True)
    spec_gt_amp_padded = pad_sequence(spec_gt_amp, batch_first=True)
    spec_gt_dop_padded = pad_sequence(spec_gt_dop, batch_first=True)
    gt_padded = pad_sequence(gt, batch_first=True)
    phonemes_padded = pad_sequence(phonemes, batch_first=True)

    return {
        'unvoiced': {
            'lp': spec_train_lp_padded,
            'pha': spec_train_pha_padded,
            'amp': spec_train_amp_padded,
            'dop': spec_train_dop_padded,
            'lengths': len_spec_train
        },
        'voiced': {
            'lp': spec_gt_lp_padded,
            'pha': spec_gt_pha_padded,
            'amp': spec_gt_amp_padded,
            'dop': spec_gt_dop_padded,
            'lengths': len_spec_gt
        },
        'gt': {
            'data': gt_padded,
            'lengths': len_gt
        },
        'phoneme': {
            'data': phonemes_padded,
            'lengths': len_phonemes
        },
        'transcript': [item['transcript'] for item in batch]
    }

class SSIDataset(Dataset):
    VALID_SPLITS = ('train', 'val', 'test')

    def __init__(self, hyperparameters: Hyperparameters, split: str = 'train', debug: bool = False) -> None:
        super().__init__()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")
        self.hyperparameters = hyperparameters
        self.split = split
        self.debug = debug
        self.data: list[tuple[tuple[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]] = []
        self.gt: list[Tensor] = []
        self.phoneme: list[Tensor] = []
        self.transcript: list[str] = []

        self.spectrogram = Spectrogram(n_fft=self.hyperparameters.n_fft, hop_length=self.hyperparameters.hop_length, power=2)
        self.ultrasr_extractor = UltraSRFeatureExtractor()
        self.lp_features = ResidualFeatureExtractor()

    def __len__(self) -> int:
        return self.data.__len__()
    
    def __getitem__(self, idx) -> dict[str, Tensor | tuple[Tensor, Tensor]]:
        return {
            'training_data': self.data[idx],
            'gt': self.gt[idx],
            'phoneme': self.phoneme[idx],
            'transcript': self.transcript[idx],
        }
    
    @staticmethod
    def _collect_valid_files(voiced_path: Path, unvoiced_path: Path, textgrid_path: Path) -> list[str]:
        """Collect all valid .flac filenames that have corresponding unvoiced and TextGrid files."""
        valid_files = []
        for file in voiced_path.glob('*.flac'):
            if unvoiced_path.joinpath(file.name).exists() and textgrid_path.joinpath(file.with_suffix('.TextGrid').name).exists():
                valid_files.append(file.name)
        return valid_files

    @staticmethod
    def _split_files(file_names: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[str]]:
        """Deterministically split file names into train/val/test lists."""
        random.seed(seed)
        shuffled = sorted(file_names)  # sort first for reproducibility
        random.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        return {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }
    
    def _load_waveform(self, path: Path, sample_rate: int) -> np.ndarray:
        waveform, r = sf.read(path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # or waveform[:, 0]
        if sample_rate != self.hyperparameters.sample_rate:
            waveform = _resample_to_target_sr(waveform, sample_rate, self.hyperparameters.sample_rate)
        return waveform

    def _load_data(self, voiced_path: Path, unvoiced_path: Path, textgrid_path: Path) -> None:
        valid_files = self._collect_valid_files(voiced_path, unvoiced_path, textgrid_path)
        split_map = self._split_files(
            valid_files,
            self.hyperparameters.train_split,
            self.hyperparameters.val_split,
            self.hyperparameters.seed
        )
        files_to_process = split_map[self.split]

        if self.debug:
            split_sizes = {k: len(v) for k, v in split_map.items()}
            print(f"Dataset split sizes: {split_sizes}")
            print(f"'{self.split}' split: {len(files_to_process)} files")

        for file_name in files_to_process:
            self._process_file(
                voiced_file=voiced_path / file_name,
                unvoiced_file=unvoiced_path / file_name,
                textgrid_file=textgrid_path / Path(file_name).with_suffix('.TextGrid'),
            )

    def load_files(self, voiced_path: Path, unvoiced_path: Path, textgrid_path: Path, file_names: list[str]) -> None:
        """Load a fixed list of recordings (no train/val/test split)."""
        for file_name in file_names:
            voiced_file = voiced_path / file_name
            unvoiced_file = unvoiced_path / file_name
            textgrid_file = textgrid_path / Path(file_name).with_suffix('.TextGrid')

            if not (voiced_file.exists() and unvoiced_file.exists() and textgrid_file.exists()):
                print(f"[dataset] Skipping {file_name}: missing voiced/unvoiced/TextGrid file in the provided directories.")
                continue

            self._process_file(voiced_file=voiced_file, unvoiced_file=unvoiced_file, textgrid_file=textgrid_file)

    def _process_file(self, voiced_file: Path, unvoiced_file: Path, textgrid_file: Path) -> None:
        """Load and extract all features for a single (voiced, unvoiced) recording pair."""
        # Load audio for lp_features
        unvoiced_waveform = self._load_waveform(unvoiced_file, self.hyperparameters.sample_rate)
        voiced_waveform = self._load_waveform(voiced_file, self.hyperparameters.sample_rate)

        # Extract lp_features
        unvoiced_lp_features = self.lp_features.forward(unvoiced_waveform)
        voiced_lp_features = self.lp_features.forward(voiced_waveform)

        unvoiced_lp_features = tensor(unvoiced_lp_features, dtype=torch.float32)
        voiced_lp_features = tensor(voiced_lp_features, dtype=torch.float32)

        unvoiced_pha, unvoiced_amp, unvoiced_dop, _, _ = load_ultrasr_features(
            unvoiced_file, self.ultrasr_extractor, auto_detect_carrier=True)
        voiced_pha, voiced_amp, voiced_dop, _, _ = load_ultrasr_features(
            voiced_file, self.ultrasr_extractor, auto_detect_carrier=True)

        # Align all features to the minimum common time dimension.
        min_time = min(
            unvoiced_lp_features.shape[0], unvoiced_pha.shape[-1], unvoiced_amp.shape[-1], unvoiced_dop.shape[-1],
            voiced_lp_features.shape[0],   voiced_pha.shape[-1],   voiced_amp.shape[-1],   voiced_dop.shape[-1],
        )
        unvoiced_lp_features  = unvoiced_lp_features[:min_time, :]
        unvoiced_pha = unvoiced_pha[..., :min_time]
        unvoiced_amp = unvoiced_amp[..., :min_time]
        unvoiced_dop = unvoiced_dop[..., :min_time]
        voiced_lp_features = voiced_lp_features[:min_time, :]
        voiced_pha = voiced_pha[..., :min_time]
        voiced_amp = voiced_amp[..., :min_time]
        voiced_dop = voiced_dop[..., :min_time]

        self.data.append(((unvoiced_lp_features, unvoiced_pha.squeeze(0).T, unvoiced_amp.squeeze(0).T, unvoiced_dop.squeeze(0).T), (voiced_lp_features, voiced_pha.squeeze(0).T, voiced_amp.squeeze(0).T, voiced_dop.squeeze(0).T)))

        voiced_log_mel = load_audio(voiced_file)
        phoneme_seq = textgrid_to_phoneme_sequence(textgrid_file, voiced_log_mel, 44100)
        self.phoneme.append(phoneme_seq)
        self.gt.append(voiced_log_mel)
        self.transcript.append(textgrid_to_transcript(textgrid_file))