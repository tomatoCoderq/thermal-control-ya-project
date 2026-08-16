import torch
import pytest

from segformer.metrics import metrics_from_counts
from segformer.model import build_segformer
from segformer.splits import split_video_ids


def test_video_split_has_no_leakage():
    ids = [f"video_{i}" for i in range(20)]
    split = split_video_ids(ids, seed=7)
    groups = [set(split.train), set(split.val), set(split.test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == set(ids)


def test_binary_metrics():
    result = metrics_from_counts(tp=8, fp=2, fn=2, tn=20)
    assert result["dice"] == pytest.approx(0.8)
    assert result["iou"] == pytest.approx(8 / 12)


def test_segformer_accepts_three_channel_square_input():
    model = build_segformer(pretrained=False)
    with torch.no_grad():
        logits = model(pixel_values=torch.randn(1, 3, 64, 64)).logits
    assert logits.shape[:2] == (1, 2)
