# thermal-control

Оценка дефектов по ИК-термографии: сегментация масок и оценка глубины залегания.
На ветке `features_unet` — модели в `models/`, единый загрузчик `TermoDataset` в
`datasets/`, эксперименты в `experiments/notebooks/`.

## Структура

```
datasets/              слой данных: TermoDataset + manifest.yaml (см. datasets/README.md)
models/                U-Net, Attention U-Net, Mamba-UNet, ConvLSTM, Thermal-Contrast
  common/              общие метрики, split, train loop
irt_data/              кэш и feature-пайплайн для TSR/Фурье (legacy yaml)
experiments/notebooks/ ноутбуки экспериментов
scripts/               препроцессинг, кэш, маски
runs/                  логи обучения (не в git)
docs/                  документация (см. docs/ARCHITECTURE.md)
```

## Окружение (uv)

Python ≥3.10. Зависимости — через [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install torch scikit-learn pyyaml pillow opencv-python matplotlib pandas
uv pip install "albumentations>=2,<3"   # для irt_data аугментаций
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

Для feature-моделей (TSR/Фурье) дополнительно нужен кэш:

```bash
python scripts/build_irt_cache.py --yaml models/U-Net/dataset_tsr.yaml
```

## TermoDataset (сырые кадры)

```python
from datasets import TermoDataset

ds = TermoDataset(
    root_dir="datasets/datasets_list",
    include=["dataset_tpu"],
)
data, mask = ds[0]   # (C, 256, 256), (256, 256)
```

Пример с аугментациями — [datasets/example.py](datasets/example.py).

## Обучение сегментации

```bash
cd models/U-Net
python train.py --yaml dataset_tsr.yaml --variants tsr --epochs 50 --device mps
```

Обзор моделей — [models/README.md](models/README.md).
