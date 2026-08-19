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

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = Path(__file__).resolve().parent / "config.yaml"


def _resolve(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


class PathsConfig(BaseModel):
    """Paths to datasets """
    kaggle_dir: Path = Path("datasets/dataset_kaggle")
    tpu_dir: Path = Path("datasets/dataset_tpu")
    mask_type: str = "automated_mask"
    feature_dir: Path = Path("features_p5")
    log_dir: Path = Path("runs")

    @field_validator("kaggle_dir", "tpu_dir", "feature_dir", "log_dir")
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


class ClassesConfig(BaseModel):
    """Classes of defects' masks"""
    gray2cls: dict[int, int] = {0: 0, 51: 1, 102: 2, 153: 3, 204: 4, 255: 5}
    cls_depth: dict[int, float] = {
        1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}  # in cm
    class_names: list[str] = ["фон", "0.5", "1.0", "1.5", "2.0", "2.5"]

    @computed_field
    @property
    def n_classes(self) -> int:
        return len(self.class_names)


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


class TrainConfig(BaseModel):
    """Параметры обучения."""

    batch_size: PositiveInt = 64
    epochs: PositiveInt = 30
    learning_rate: PositiveFloat = 1e-3
    n_test_videos: PositiveInt = 10

    max_videos: Optional[int] = None
    # ce | weighted_ce | label_smooth
    loss_name: str = "ce"
    seed: int = 67


class Config(BaseModel):
    """Main config with all subconfigs"""

    paths: PathsConfig = PathsConfig()
    classes: ClassesConfig = ClassesConfig()
    tsr: TSRConfig = TSRConfig()
    crop: CropConfig = CropConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

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
    # Быстрая проверка + генерация дефолтного config.yaml, если его нет/он пуст.
    cfg = load_config()
    print("ROOT        :", ROOT)
    print("data_dir    :", cfg.paths.data_dir)
    print("mask_dir    :", cfg.paths.mask_dir)
    print("n_classes   :", cfg.classes.n_classes)
    print("n_channels  :", cfg.tsr.n_channels)
    print("train       :", cfg.train.model_dump())
    if not DEFAULT_YAML.exists() or DEFAULT_YAML.stat().st_size == 0:
        Config().dump_yaml(DEFAULT_YAML)
        print("записан дефолтный", DEFAULT_YAML)
