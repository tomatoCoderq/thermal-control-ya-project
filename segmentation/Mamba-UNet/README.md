# Mamba-UNet (thermal IRT)

Adapted from nearby [`Mamba-UNet`](../../../Mamba-UNet) (VSSM / VMamba U-Net).

## Task settings

| | |
|---|---|
| Input size | **256×256** |
| TSR (`tsr_coeffs`, poly_degree=5) | **6 channels** |
| Fourier (`ppt`, bins 1–6) | **6 channels** |
| Output | 1-channel logits (binary mask) |
| Loss | BCE + Soft Dice |

Dataset YAML is shared with `segmentation/U-Net/` (`dataset.yaml`).

## Run

```bash
cd segmentation/Mamba-UNet
python train.py --variants tsr --epochs 50
python train.py --variants fourier --epochs 50
python eval.py

# curves (same format as U-Net / Attention)
python -m common.plot_history runs/tsr --save runs/tsr/curves.png
```

Logs: `runs/<variant>/history.json` + `metrics.csv` (train_* / test_*).

## Notes

- Backbone: `mamba_sys.VSSM` with `depths=[2,2,2,2]`, `patch_size=4`, `in_chans=6`.
- If `mamba_ssm` CUDA kernels are installed, they are used; otherwise a pure-PyTorch selective scan runs (CPU/MPS OK, but slower — keep batch ≤4).
- Original medical SSL configs (224², multi-class) are not used; no pretrained ImageNet/VMAMBA weights by default.
