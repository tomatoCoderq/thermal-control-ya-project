# Legacy Attention U-Net (nn.Upsample decoder)

Старые чекпоинты и история обучения до перехода на `ConvTranspose2d` (MPS-safe).

## Содержимое

- `checkpoints/` — `model_attunet_{tsr,pca}_{best,last}*.tar`
- `runs/` — `history.json`, `metrics.csv` по вариантам
- `model.py` — архитектура с `nn.Upsample` (нужна для resume)

## Продолжить старое обучение

```bash
cd models/Attention\ U-Net
/Users/user/Education/CVYandexCamp/venv/bin/python legacy_upsample/train.py \
  --variants tsr --epochs 25 --resume best --device cpu
```

На MPS backward у legacy-модели может падать — предпочитай `--device cpu` или Colab CUDA.

Новое обучение — только через корневой `train.py` + `model.py` (новая архитектура).
