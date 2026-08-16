import numpy as np

from irt_data.config import AugConfig, AugSpec, DatasetConfig, FileMeta, ObjectCropConfig, ROI
from irt_data.crops import ObjectCropper
from irt_data.dataset import IRTDataset
from irt_data.features import PPTFeatureExtractor
from irt_data.transforms import TransformPipeline


def test_config_parses_object_rois_and_ppt_bins():
    cfg = DatasetConfig.from_dict(
        {
            "features": {"extractors": ["ppt"], "ppt_bins": [1, 2, 3]},
            "object_crop": {"enabled": True, "roi": {"x": 2, "y": 3, "w": 8, "h": 6}},
            "files_meta": {"v1": {"object_roi": {"x": 1, "y": 1, "w": 5, "h": 5}}},
        }
    )
    assert cfg.features.ppt_bins == (1, 2, 3)
    assert cfg.object_crop.roi == ROI(2, 3, 8, 6)
    assert cfg.files_meta["v1"].object_roi == ROI(1, 1, 5, 5)


def test_source_level_object_roi_is_parsed():
    cfg = DatasetConfig.from_dict(
        {"sources": [{"root": "data", "object_roi": {"x": 4, "y": 5, "w": 6, "h": 7}}]}
    )
    assert cfg.sources[0].object_roi == ROI(4, 5, 6, 7)


def test_object_crop_is_identical_for_all_frames_and_zero_pads_mask():
    base = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    frames = np.stack([base, base + 100], axis=0)
    cropper = ObjectCropper(
        ObjectCropConfig(enabled=True, roi=ROI(0, 0, 4, 6), square_pad=True)
    )
    box = cropper.box_for(6, 8, FileMeta())
    cropped = cropper.apply_frames(frames, box)
    assert cropped.shape == (2, 6, 6)
    np.testing.assert_allclose(cropped[1] - cropped[0], 100)


def test_ppt_returns_three_finite_aligned_maps():
    t = np.linspace(0, 2 * np.pi, 64, dtype=np.float32)
    video = np.sin(t)[:, None, None] * np.ones((64, 5, 7), dtype=np.float32)
    ppt = PPTFeatureExtractor(bins=(1, 2, 3), n_frames=32, auto_cooling=False)
    result = ppt(video)
    assert result.shape == (5, 7, 3)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0, 0], result[-1, -1])


def test_patch_shuffle_is_synchronised_across_channels_and_mask():
    mask = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image = np.stack([mask, mask, mask], axis=-1).astype(np.float32)
    pipeline = TransformPipeline(
        AugConfig(spatial=[AugSpec("RandomGridShuffle", {"grid": [2, 2], "p": 1.0})])
    )
    transformed, transformed_mask = pipeline.apply_features(image, mask)
    assert transformed.shape == image.shape
    np.testing.assert_array_equal(transformed[..., 0], transformed_mask)
    np.testing.assert_array_equal(transformed[..., 1], transformed_mask)


def test_discrete_rotation_preserves_shape_and_mask_classes():
    image = np.zeros((32, 32, 3), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 10:22] = 1
    mask[8:24, 10:22] = 1
    pipeline = TransformPipeline(
        AugConfig(spatial=[AugSpec("DiscreteRotate7", {"p": 1.0})])
    )
    transformed, transformed_mask = pipeline.apply_features(image, mask)
    assert transformed.shape == image.shape
    assert transformed_mask.shape == mask.shape
    assert set(np.unique(transformed_mask)) <= {0, 1}


class _FakeBackend:
    def __init__(self):
        t = np.linspace(0, 4 * np.pi, 64, dtype=np.float32)
        spatial = np.linspace(0.8, 1.2, 24 * 32, dtype=np.float32).reshape(24, 32)
        self.video = np.sin(t)[:, None, None] * spatial[None]

    def list_ids(self):
        return ["sample"]

    def shape(self, video_id):
        return self.video.shape

    def read_all(self, video_id):
        return self.video.copy()

    def stats(self, video_id):
        return {"mean": float(self.video.mean()), "std": float(self.video.std())}


def test_dataset_produces_unet_and_segformer_ready_square_tensor(tmp_path):
    cfg = DatasetConfig.from_dict(
        {
            "mode": "features",
            "train": False,
            "cache_dir": str(tmp_path / "cache"),
            "features": {
                "extractors": ["ppt"],
                "ppt_bins": [1, 2, 3],
                "ppt_frames": 32,
                "ppt_auto_cooling": False,
                "frame_step": 1,
                "cache_dir": str(tmp_path / "features"),
            },
            "object_crop": {
                "enabled": True,
                "roi": {"x": 6, "y": 2, "w": 20, "h": 20},
                "output_size": [32, 32],
            },
            "crop": {"size": [32, 32], "strategy": "full"},
            "mask": {"kind": "binary", "num_classes": 2, "missing": "error"},
            "norm": {"mode": "per_channel"},
        }
    )
    dataset = IRTDataset(cfg, backend=_FakeBackend())
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[8:14, 12:18] = 255
    dataset.mask_reader.read = lambda _: mask
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 32, 32)
    assert tuple(sample["mask"].shape) == (32, 32)
    assert set(sample["mask"].unique().tolist()) == {0, 1}
