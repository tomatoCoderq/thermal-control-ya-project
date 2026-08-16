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
    feature_dir: Path = Path("features_p5")          # kaggle TSR-признаки
    tpu_feature_dir: Path = Path("features_tpu")     # tpu TSR-признаки (256x320)
    log_dir: Path = Path("runs")

    @field_validator("kaggle_dir", "tpu_dir", "feature_dir",
                     "tpu_feature_dir", "log_dir")
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
        """Маски глубины tpu (мм, фон=NaN) из make_tpu_masks.py."""
        return self.tpu_dir / "labels" / "table_mask"


class ClassesConfig(BaseModel):
    """Classes of defects' masks.

    Для регрессии глубины держим единицы в МИЛЛИМЕТРАХ (общий формат с tpu).
    `cls_depth` остаётся в см (совместимость с 5p), а `cls_depth_mm` = ×10.
    Классы нужны только для (а) построения kaggle-таргета и (б) дискретизации
    предсказаний обратно в классы ради сравнимых accuracy/QWK.
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

    Таргет — глубина залегания дефекта в мм. Для устойчивости обучения таргет
    z-score-нормируется по train (`target_mean`/`target_std` заполняются на этапе
    сборки данных и снапшотятся в конфиг прогона); лосс считается в нормированном
    пространстве, метрики денормируются обратно в мм.
    """

    depth_unit: str = "mm"
    # smooth_l1 (Huber) | l1 | mse
    loss_name: str = "smooth_l1"
    huber_beta: PositiveFloat = 1.0          # порог Huber, в НОРМИРОВАННЫХ единицах
    # статистики таргета (мм); None → посчитать по train и записать в снапшот
    target_mean: Optional[float] = None
    target_std: Optional[float] = None


class OptimizerConfig(BaseModel):
    """Оптимизатор: adam | adamw | muon (гибрид Muon + Adam).

    Muon применяется к 2D+ скрытым весам, Adam — к bias/нормам/голове. У групп
    разные lr: скрытые матрицы (muon_lr) обучаются заметно «крупнее», чем adam_lr.
    """
    name: str = "adam"                      # adam | adamw | muon
    weight_decay: float = Field(0.0, ge=0.0)
    # muon-специфичное
    muon_lr: PositiveFloat = 0.02
    adam_lr: PositiveFloat = 3e-4           # lr adam-группы в гибриде muon
    momentum: float = Field(0.95, ge=0.0, lt=1.0)
    nesterov: bool = True
    ns_steps: PositiveInt = 5               # итераций Ньютона–Шульца


class TrainConfig(BaseModel):
    """Параметры обучения."""

    # classification | regression — какую голову/лосс/метрики использовать
    task: str = "regression"

    batch_size: PositiveInt = 64
    epochs: PositiveInt = 30
    learning_rate: PositiveFloat = 1e-3
    n_test_videos: PositiveInt = 10

    max_videos: Optional[int] = None
    # (для классификации) ce | weighted_ce | label_smooth
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
