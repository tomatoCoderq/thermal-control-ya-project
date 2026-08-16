"""IRT NDT dataset package: features and temporal modes for semantic segmentation."""

__all__ = ["DatasetConfig", "IRTDataset", "build_dataloader"]


def __getattr__(name: str):
    if name == "DatasetConfig":
        from irt_data.config import DatasetConfig

        return DatasetConfig
    if name == "IRTDataset":
        from irt_data.dataset import IRTDataset

        return IRTDataset
    if name == "build_dataloader":
        from irt_data.loaders import build_dataloader

        return build_dataloader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
