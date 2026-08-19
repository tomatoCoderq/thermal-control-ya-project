from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    PositiveFloat,
    computed_field,
    field_validator,
)

# experiments/regression/config.py → repo root = parents[2]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = Path(__file__).resolve().parent / "config.yaml"


def _resolve(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


class PathsConfig(BaseModel):
    """Пути. Сырые данные и маски теперь берутся через TermoDataset из
    `datasets_root` (см. main.iter_samples); feature_dir/tpu_feature_dir —
    кэши TSR-признаков, log_dir — логи прогонов."""
    datasets_root: Path = Path("datasets/datasets_list")
    feature_dir: Path = Path("features_p5")          # kaggle TSR-признаки
    tpu_feature_dir: Path = Path("features_tpu")     # tpu TSR-признаки
    log_dir: Path = Path("runs")

    @field_validator("datasets_root", "feature_dir", "tpu_feature_dir", "log_dir")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return _resolve(v)


class ClassesConfig(BaseModel):
    """Классы дефектов kaggle-масок.

    Для регрессии глубины держим единицы в МИЛЛИМЕТРАХ (общий формат с tpu).
    `cls_depth` в см (совместимость с 5p), `cls_depth_mm` = ×10. Классы нужны для
    (а) построения kaggle-таргета и (б) дискретизации предсказаний ради
    сравнимых accuracy/QWK.
    """
    gray2cls: dict[int, int] = {0: 0, 51: 1, 102: 2, 153: 3, 204: 4, 255: 5}
    cls_depth: dict[int, float] = {
        1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}  # in cm
    class_names: list[str] = ["фон", "0.5", "1.0", "1.5", "2.0", "2.5"]

    @computed_field
    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @computed_field
    @property
    def cls_depth_mm(self) -> dict[int, float]:
        """Класс → глубина в мм (kaggle: см ×10)."""
        return {c: d * 10.0 for c, d in self.cls_depth.items()}

    @computed_field
    @property
    def depth_bins_mm(self) -> list[float]:
        """Отсортированные kaggle-глубины (мм) — узлы для дискретизации pred→класс."""
        return [self.cls_depth_mm[c] for c in sorted(self.cls_depth_mm)]


class TPUConfig(BaseModel):
    """tpu-маски png кодируют глубину линейно: gray = round(255*depth/depth_max),
    поэтому depth_mm = gray * depth_max / 255 (см. scripts/make_tpu_masks.py)."""
    depth_max_mm: PositiveFloat = 6.1


class TSRConfig(BaseModel):
    """TSR preproc data"""

    poly_degree: PositiveInt = 5

    @computed_field
    @property
    def n_channels(self) -> int:
        return self.poly_degree + 1


class CropConfig(BaseModel):
    crop: PositiveInt = 48  # size of crop

    # these are background per video (not defects!!)
    n_bg_per_video: int = Field(8, ge=0)


class ModelConfig(BaseModel):
    name: str = "small_cnn"
    dropout: float = Field(0.3, ge=0.0, le=1.0)
    pretrained: bool = True  # only for downloaded models


class RegressionConfig(BaseModel):
    """Регрессия глубины (вариант B: чистая регрессия, 1 выход).

    Таргет — глубина залегания в мм; z-score-нормируется по train
    (`target_mean`/`target_std`), лосс в нормированном пространстве, метрики
    денормируются в мм.
    """

    depth_unit: str = "mm"
    # smooth_l1 (Huber) | l1 | mse
    loss_name: str = "smooth_l1"
    huber_beta: PositiveFloat = 1.0          # порог Huber, в НОРМИРОВАННЫХ единицах
    target_mean: Optional[float] = None
    target_std: Optional[float] = None


class OptimizerConfig(BaseModel):
    """Оптимизатор: adam | adamw | muon (гибрид Muon + Adam)."""
    name: str = "adam"                      # adam | adamw | muon
    weight_decay: float = Field(0.0, ge=0.0)
    muon_lr: PositiveFloat = 0.02
    adam_lr: PositiveFloat = 3e-4
    momentum: float = Field(0.95, ge=0.0, lt=1.0)
    nesterov: bool = True
    ns_steps: PositiveInt = 5


class TrainConfig(BaseModel):
    """Параметры обучения."""

    task: str = "regression"

    batch_size: PositiveInt = 64
    epochs: PositiveInt = 30
    learning_rate: PositiveFloat = 1e-3
    n_test_videos: PositiveInt = 10

    max_videos: Optional[int] = None
    loss_name: str = "ce"
    seed: int = 67


class Config(BaseModel):
    """Main config with all subconfigs"""

    paths: PathsConfig = PathsConfig()
    classes: ClassesConfig = ClassesConfig()
    tpu: TPUConfig = TPUConfig()
    tsr: TSRConfig = TSRConfig()
    crop: CropConfig = CropConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    optim: OptimizerConfig = OptimizerConfig()
    regression: RegressionConfig = RegressionConfig()

    def dump_yaml(self, path: str | Path) -> None:
        """to yaml file"""
        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_config(path: str | Path = DEFAULT_YAML) -> Config:
    path = Path(path)
    if not path.exists():
        print("Empty/Standard config")
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)


# global config
CFG = load_config()

if __name__ == "__main__":
    cfg = load_config()
    print("ROOT          :", ROOT)
    print("datasets_root :", cfg.paths.datasets_root)
    print("feature_dir   :", cfg.paths.feature_dir)
    print("n_classes     :", cfg.classes.n_classes)
    print("n_channels    :", cfg.tsr.n_channels)
