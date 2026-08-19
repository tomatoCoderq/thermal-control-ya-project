# Segmentation

Единственный рабочий пайплайн сегментации — **Thermal-Contrast** (Heat Control):

```
TermoDataset (T, H, W)  →  extract_channels  →  (4, H, W)  →  U-Net  →  маска
```

## Структура

```
models/
  Thermal-Contrast/   # train / eval / inference / каналы / чекпоинты
  U-Net/main.py        # архитектура U-Net (импортируется из Thermal-Contrast)
  common/              # device, loop, metrics, split, tracking
```

## Быстрый старт

```bash
cd models/Thermal-Contrast
python train.py --epochs 50
python eval.py
```

Или ноутбук: `experiments/notebooks/contrast_train.ipynb`.

Подробности — в [Thermal-Contrast/README.md](Thermal-Contrast/README.md).
