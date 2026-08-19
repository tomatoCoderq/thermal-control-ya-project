import yaml
from dataclasses import dataclass


@dataclass
class CropConfig:
    x0: int | None
    x1: int | None
    y0: int | None
    y1: int | None

@dataclass
class DataConfig:
    path: str
    file_pattern: str
    mat_key: str
    dtype: str

@dataclass
class MaskConfig:
    path: str
    file_pattern: str

@dataclass
class DatasetConfig:
    name: str
    data: DataConfig
    masks: MaskConfig
    crop: CropConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw["name"],
            data=DataConfig(**raw["data"]),
            masks=MaskConfig(**raw["masks"]),
            crop=CropConfig(**raw["crop"]),
        )