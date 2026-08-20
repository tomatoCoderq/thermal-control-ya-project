# thermal-control

Оценка дефектов по ИК-термографии: сегментация маски и глубина залегания
каждого найденного экземпляра. Сырое `.mat`-видео сжимается общими
трансформами в многоканальную карту; U-Net даёт маску; по кропам вокруг
связных компонент отдельная сеть предсказывает глубину в мм.

Подробно — [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Структура

```
datasets/          TermoDataset + манифесты + трансформы (train = inference)
models/            U-Net сегментации, CNN регрессии, Lightning-обёртки
pipeline/          ThermalControlVideoPredictor → mask + depth
web-frontend/      Gradio: .mat → overlay, таблица дефектов, файлы
experiments/       исторические прогоны (lstm, TSR, FFT) — не продакшен-путь
scripts/           маски TPU, TIFF/видео, превью признаков
docs/              архитектура, планы, вопросы
```

## Окружение (uv)

Python ≥ 3.10. Зависимости и виртуальное окружение — через [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install torch pytorch-lightning timm scikit-learn pyyaml pillow opencv-python matplotlib pandas
```

## Данные

Наборы не хранятся в git. Каждый поддатасет самоописывается своим `manifest.yaml`;
добавить новый набор — положить папку с `data/`, `masks/` и манифестом.

```
datasets/datasets_list/
  dataset_kaggle/   manifest.yaml + data/*.mat + masks/*.png   # mat_key: imageArray
  dataset_tpu/      manifest.yaml + data/*.mat + masks/*.png   # mat_key: data
```

Kaggle-набор: <https://www.kaggle.com/datasets/ziangwei/irt-pvc-depth>

## Данные → тензор

```python
from datasets.datasets import TermoDataset
from datasets.transforms import (
    Compose, Stack, SelectFrames, PercentileNorm,
    MaxMin, Std, PCA1, TSRDeriv,
)

tf = Compose([
    SelectFrames(num_frames=128),
    Stack([MaxMin(), Std(), PCA1(), TSRDeriv(1), TSRDeriv(2)]),  # → 5 ch
    PercentileNorm(),
])

ds = TermoDataset(
    root_dir="datasets/datasets_list",
    include=["dataset_tpu"],   # None — все поддатасеты
    transform=tf,
)

data, mask = ds[0]             # (5, 256, 256), (256, 256)
```

Тот же `Compose` подаётся в `ThermalControlVideoPredictor` как `seg_transform`.
Пример с аугментациями — [datasets/example.py](datasets/example.py).
Описание загрузчика — [datasets/README.md](datasets/README.md).

## Инференс

Два каскада: сегментация полного кадра → кропы 48×48 → регрессия глубины.
См. раздел «Как пользоваться» в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Gradio:

```bash
THERMAL_SEG_CKPT=path/to/seg.pkl THERMAL_REG_CKPT=path/to/reg.pkl \
  python web-frontend/app.py
```
