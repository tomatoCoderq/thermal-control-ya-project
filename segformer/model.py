"""SegFormer model construction without importing classification code."""

from __future__ import annotations


def build_segformer(
    model_name: str = "nvidia/mit-b0",
    num_labels: int = 2,
    pretrained: bool = True,
):
    """Build a Hugging Face SegFormer with a binary segmentation head."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    id2label = {0: "background", 1: "defect"}
    label2id = {v: k for k, v in id2label.items()}
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    config = SegformerConfig(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        num_channels=3,
    )
    return SegformerForSemanticSegmentation(config)
