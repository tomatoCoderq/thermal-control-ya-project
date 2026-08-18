# Thermal-Contrast U-Net (Variant A — static baseline)

Свернуть **клип IRT-кадров** в один информативный 2D-тензор, затем обычный U-Net.
Без ConvLSTM / без TSR-коэффициентов на весь ролик — только локальная temporal-сводка.

## Идея

```
(T, H, W) clip  →  collapse  →  (C, H, W) contrast maps  →  U-Net  →  mask
```

| Канал | Формула | Смысл |
|-------|---------|--------|
| `maxmin` | max(t) − min(t) | тепловой контраст (классика NDT) |
| `maxfirst` | max(t) − frame₀ | пик нагрева vs начало |
| `std` | σ по времени | нестабильность / динамика |
| `pca1` | 1-я PCA по t | доминирующий temporal mode |

## Presets

| preset | Каналы | in_ch |
|--------|--------|-------|
| `minimal` | maxmin | 1 |
| `delta` | maxmin, maxfirst | 2 |
| `combo` | maxmin, maxfirst, std | 3 **(default)** |
| `pca` | maxmin, pca1 | 2 |
| `full` | все четыре | 4 |

## Данные

Тот же yaml, что ConvLSTM: kaggle + TPU, `mode: temporal`, 12 кадров.

```bash
python scripts/build_irt_cache.py --yaml segmentation/Thermal-Contrast/dataset_contrast.yaml
```

## Train

```bash
cd segmentation/Thermal-Contrast
export NO_ALBUMENTATIONS_UPDATE=1

python train.py --preset combo --device mps --epochs 50 --workers 0
python train.py --preset minimal --epochs 50   # только max-min
python train.py --preset pca --epochs 50      # max-min + temporal PCA
```

Чекпоинты: `model_contrast_{preset}_best.tar`  
Логи: `runs/{preset}/history.json`

**Ноутбук:** `notebooks/contrast_train.ipynb` — свёртка клипа, live-кривые, превью каждые N эпох.

## Eval

```bash
python eval.py
```

## vs ConvLSTM / TSR

| | Thermal-Contrast | ConvLSTM | TSR U-Net |
|--|------------------|----------|-----------|
| Temporal | свёрнут в 2D | recurrent | poly fit на весь ролик |
| Модель | U-Net | ConvLSTM+decoder | U-Net |
| Скорость infer | быстро | медленнее (T steps) | быстро (offline features) |

Хороший **baseline** перед ConvLSTM: если contrast U-Net уже близок к TSR — temporal свёртка достаточна; если нет — ConvLSTM может выиграть за счёт явной памяти.
