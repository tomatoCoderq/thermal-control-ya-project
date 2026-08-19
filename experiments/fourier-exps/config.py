from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, computed_field, field_validator


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = Path(__file__).resolve().parent / "config.yaml"


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    labels_dir: Path = Path("labels")
    mask_type: str = "automated_mask"
    feature_dir: Path = Path("fourier_features")
    log_dir: Path = Path("fourier_runs")

    @field_validator("data_dir", "labels_dir", "feature_dir", "log_dir")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return _resolve(value)

    @computed_field
    @property
    def mask_dir(self) -> Path:
        return self.labels_dir / self.mask_type


class ClassesConfig(BaseModel):
    gray2cls: dict[int, int] = {0: 0, 51: 1, 102: 2, 153: 3, 204: 4, 255: 5}
    cls_depth: dict[int, float] = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}
    class_names: list[str] = ["фон", "0.5", "1.0", "1.5", "2.0", "2.5"]

    @computed_field
    @property
    def n_classes(self) -> int:
        return len(self.class_names)


class FourierConfig(BaseModel):
    """Параметры pulsed phase thermography (FFT выполняется по времени)."""

    first_bin: PositiveInt = 1
    n_frequencies: PositiveInt = 8
    phase_encoding: Literal["raw", "sin_cos"] = "raw"
    window: Literal["none", "hann"] = "none"
    detrend: Literal["none", "constant", "linear"] = "none"
    start_at_peak: bool = False
    row_chunk: PositiveInt = 32

    @computed_field
    @property
    def n_channels(self) -> int:
        multiplier = 2 if self.phase_encoding == "sin_cos" else 1
        return self.n_frequencies * multiplier


class CropConfig(BaseModel):
    crop: PositiveInt = 48
    n_bg_per_video: int = Field(8, ge=0)


class ModelConfig(BaseModel):
    name: str = "convnext_nano"
    dropout: float = Field(0.3, ge=0.0, le=1.0)
    pretrained: bool = False


class TrainConfig(BaseModel):
    batch_size: PositiveInt = 64
    epochs: PositiveInt = 50
    learning_rate: PositiveFloat = 1e-3
    n_test_videos: PositiveInt = 10
    max_videos: Optional[int] = None
    loss_name: str = "label_smooth"
    seed: int = 67


class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    classes: ClassesConfig = ClassesConfig()
    fourier: FourierConfig = FourierConfig()
    crop: CropConfig = CropConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def dump_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                self.model_dump(mode="json"), stream, allow_unicode=True, sort_keys=False
            )


def load_config(path: str | Path = DEFAULT_YAML) -> Config:
    path = Path(path)
    if not path.exists():
        return Config()
    with path.open("r", encoding="utf-8") as stream:
        return Config.model_validate(yaml.safe_load(stream) or {})


CFG = load_config()




