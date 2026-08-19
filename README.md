# thermal-control

Оценка дефектов по ИК-термографии: сегментация масок через Thermal-Contrast (Heat Control).

## Структура

```
datasets/              TermoDataset + manifest.yaml (см. datasets/README.md)
models/                Thermal-Contrast + U-Net
  common/              device, metrics, split, train loop
experiments/notebooks/ contrast_train / contrast_channels_preview
scripts/               препроцессинг, маски
docs/                  документация
```

## Окружение

Python ≥3.10. Удобный venv: `/Users/user/Education/CVYandexCamp/venv`.

```bash
source /Users/user/Education/CVYandexCamp/venv/bin/activate
```

## Данные

Наборы не хранятся в git. Каждый поддатасет описывается `manifest.yaml`:

```
datasets/datasets_list/
  dataset_kaggle/   manifest.yaml + data/*.mat + masks/*.png
  dataset_tpu/      manifest.yaml + data/*.mat + masks/*.png
```

Подробности — [datasets/README.md](datasets/README.md).

Kaggle: <https://www.kaggle.com/datasets/ziangwei/irt-pvc-depth>

## TermoDataset

```python
from datasets import TermoDataset

ds = TermoDataset(root_dir="datasets/datasets_list")
data, mask = ds[0]   # (T, 256, 256), (256, 256)
```

## Обучение сегментации (Thermal-Contrast)

```bash
cd models/Thermal-Contrast
python train.py --epochs 50
python eval.py
```

Ноутбук: `experiments/notebooks/contrast_train.ipynb`.  
Обзор — [models/README.md](models/README.md).
