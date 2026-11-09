from __future__ import annotations
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from dataset.residual_features import ResidualFeatureExtractor
from dataset.transformer_dataset import SSIDataset

@dataclass(frozen=True)
class AblationSpec:
    """Which input region to zero out for a single ablation run."""

    name: str = "baseline"
    stream: str = "baseline"
    lp_indices: tuple[int, ...] = field(default_factory=tuple)
    zero_pha: bool = False
    zero_amp: bool = False
    zero_dop: bool = False

    @property
    def is_baseline(self) -> bool:
        return self.stream == "baseline"

    @property
    def run_type(self) -> str:
        return "baseline" if self.is_baseline else "ablation"

class MaskedDataset(Dataset):
    """Wraps an SSIDataset and zeroes the region(s) described by an AblationSpec."""

    def __init__(self, dataset: SSIDataset, spec: AblationSpec):
        self.dataset = dataset
        self.spec = spec

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        if self.spec.is_baseline:
            return item

        item = dict(item)
        item["training_data"] = (
            self._mask_branch(item["training_data"][0]),
            self._mask_branch(item["training_data"][1]),
        )
        return item

    def _mask_branch(self, branch) -> tuple:
        lp, pha, amp, dop = branch
        lp = lp.clone()
        pha = pha.clone()
        amp = amp.clone()
        dop = dop.clone()

        if self.spec.stream == "lp":
            for i in self.spec.lp_indices:
                if 0 <= i < lp.shape[-1]:
                    lp[:, i] = 0.0
        elif self.spec.stream == "pha":
            pha.zero_()
        elif self.spec.stream == "amp":
            amp.zero_()
        elif self.spec.stream == "dop":
            dop.zero_()

        return lp, pha, amp, dop

def build_run_specs() -> list[AblationSpec]:
    """Enumerate all runs: baseline first, then one per LP feature and UltraSR stream."""
    specs: list[AblationSpec] = [AblationSpec(name="baseline")]

    feature_names = ResidualFeatureExtractor().get_feature_names()  # 34 names
    for i, name in enumerate(feature_names):
        specs.append(AblationSpec(name=f"lp_{name}", stream="lp", lp_indices=(i,)))

    specs.append(AblationSpec(name="phase_gradient", stream="pha", zero_pha=True))
    specs.append(AblationSpec(name="amplitude_gradient", stream="amp", zero_amp=True))
    specs.append(AblationSpec(name="doppler_spectrogram", stream="dop", zero_dop=True))

    return specs