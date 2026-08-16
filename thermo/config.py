"""Единый конфиг общего модуля thermo.

Сводит воедино то, что раньше дублировалось в 5th-polynom-exps / fourier-exps /
regression-experiments. Ключевые «ручки»:
  * train.task     — classification | regression (голова/лосс/метрики/метка);
  * features.kind  — tsr | fourier (какой экстрактор признаков);
  * features.deriv — p5 | p5+d1 | p5+d1+d2 (доп. каналы-производные, общее для всех).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import (
    BaseModel, Field, PositiveInt, PositiveFloat, computed_field, field_validator,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = Path(__file__).resolve().parent / "config.yaml"


def _resolve(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


class PathsConfig(BaseModel):
    kaggle_dir: Path = Path("datasets/dataset_kaggle")
    tpu_dir: Path = Path("datasets/dataset_tpu")
    mask_type: str = "automated_mask"
    # кэш TSR-признаков (совместимость с существующими каталогами)
    tsr_kaggle_dir: Path = Path("features_p5")
    tsr_tpu_dir: Path = Path("features_tpu")
    # кэш признаков прочих экстракторов: <features_cache>/<kind>_<domain>
    features_cache: Path = Path("features_cache")
    log_dir: Path = Path("runs")

    @field_validator("kaggle_dir", "tpu_dir", "tsr_kaggle_dir", "tsr_tpu_dir",
                     "features_cache", "log_dir")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return _resolve(v)

    @computed_field
    @property
    def data_dir(self) -> Path:
        return self.kaggle_dir / "data"

    @computed_field
    @property
    def mask_dir(self) -> Path:
        return self.kaggle_dir / "labels" / self.mask_type

    @computed_field
    @property
    def tpu_mask_dir(self) -> Path:
        return self.tpu_dir / "labels" / "table_mask"

    def feature_dir(self, kind: str, domain: str) -> Path:
        """Каталог кэша признаков по (экстрактор, домен)."""
        if kind == "tsr":
            return self.tsr_kaggle_dir if domain == "kaggle" else self.tsr_tpu_dir
        return self.features_cache / f"{kind}_{domain}"


class ClassesConfig(BaseModel):
    gray2cls: dict[int, int] = {0: 0, 51: 1, 102: 2, 153: 3, 204: 4, 255: 5}
    cls_depth: dict[int, float] = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}  # см
    class_names: list[str] = ["фон", "0.5", "1.0", "1.5", "2.0", "2.5"]

    @computed_field
    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @computed_field
    @property
    def cls_depth_mm(self) -> dict[int, float]:
        return {c: d * 10.0 for c, d in self.cls_depth.items()}

    @computed_field
    @property
    def depth_bins_mm(self) -> list[float]:
        return [self.cls_depth_mm[c] for c in sorted(self.cls_depth_mm)]


class FeaturesConfig(BaseModel):
    """Экстрактор признаков и его параметры (единый для tsr/fourier)."""
    kind: str = "tsr"                 # tsr | fourier
    deriv: str = "p5"                 # p5 | p5+d1 | p5+d1+d2 (каналы-производные)

    # tsr
    poly_degree: PositiveInt = 5

    # fourier (фаза rFFT), см. thermo/features/fourier.py
    n_frequencies: PositiveInt = 8
    first_bin: PositiveInt = 1
    phase_encoding: str = "sin_cos"   # raw | sin_cos (wrap-safe)
    window: str = "none"              # none | hann
    detrend: str = "none"             # none | constant | linear
    start_at_peak: bool = False

    @computed_field
    @property
    def base_channels(self) -> int:
        """Каналов на выходе экстрактора (до производных)."""
        if self.kind == "tsr":
            return self.poly_degree + 1
        if self.kind == "fourier":
            return self.n_frequencies * (2 if self.phase_encoding == "sin_cos" else 1)
        raise ValueError(f"неизвестный features.kind: {self.kind}")

    @computed_field
    @property
    def in_channels(self) -> int:
        """Каналов на входе модели = base + производные (deriv)."""
        b = self.base_channels
        return {"p5": b, "p5+d1": b + (b - 1),
                "p5+d1+d2": b + (b - 1) + (b - 2)}[self.deriv]

    def signature(self) -> dict:
        """Параметры, влияющие на кэш признаков (для инвалидации)."""
        if self.kind == "tsr":
            return {"kind": "tsr", "poly_degree": self.poly_degree}
        return {"kind": "fourier", "n_frequencies": self.n_frequencies,
                "first_bin": self.first_bin, "phase_encoding": self.phase_encoding,
                "window": self.window, "detrend": self.detrend,
                "start_at_peak": self.start_at_peak}


class CropConfig(BaseModel):
    crop: PositiveInt = 48
    n_bg_per_video: int = Field(8, ge=0)


class ModelConfig(BaseModel):
    name: str = "small_cnn"
    dropout: float = Field(0.3, ge=0.0, le=1.0)
    pretrained: bool = False


class RegressionConfig(BaseModel):
    depth_unit: str = "mm"
    augment: bool = False
    loss_name: str = "smooth_l1"      # smooth_l1 | l1 | mse
    huber_beta: PositiveFloat = 1.0
    target_mean: Optional[float] = None
    target_std: Optional[float] = None


class OptimizerConfig(BaseModel):
    name: str = "adam"                # adam | adamw | muon
    weight_decay: float = Field(0.0, ge=0.0)
    muon_lr: PositiveFloat = 0.02
    adam_lr: PositiveFloat = 3e-4
    momentum: float = Field(0.95, ge=0.0, lt=1.0)
    nesterov: bool = True
    ns_steps: PositiveInt = 5


class TrainConfig(BaseModel):
    task: str = "regression"          # classification | regression
    batch_size: PositiveInt = 64
    epochs: PositiveInt = 30
    learning_rate: PositiveFloat = 1e-3
    n_test_videos: PositiveInt = 10
    max_videos: Optional[int] = None
    loss_name: str = "label_smooth"   # (classification) ce | weighted_ce | label_smooth
    seed: int = 67


class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    classes: ClassesConfig = ClassesConfig()
    features: FeaturesConfig = FeaturesConfig()
    crop: CropConfig = CropConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    optim: OptimizerConfig = OptimizerConfig()
    regression: RegressionConfig = RegressionConfig()

    def dump_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f,
                           allow_unicode=True, sort_keys=False)


def load_config(path: str | Path = DEFAULT_YAML) -> Config:
    path = Path(path)
    if not path.exists():
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)


CFG = load_config()
