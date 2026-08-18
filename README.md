# thermal-control

Оценка дефектов по ИК-термографии (сегментация масок / глубина залегания). На
`main` — слой данных: единый загрузчик `TermoDataset`, сводящий разнородные
термо-датасеты к одному формату `(data, mask)`. Обучение и эксперименты лежат в
ноутбуках и на ветках. Работает на нескольких наборах (kaggle PVC, tpu, CFRP) с
переносом между ними.

## Структура

```
datasets/          слой данных: TermoDataset, конфиг из manifest.yaml (см. datasets/README.md)
scripts/           препроцессинг: маски глубины, экспорт в TIFF/видео
experiments/       ноутбуки экспериментов
thermo_deprecated/ прежний унифицированный пайплайн (не развивается)
runs/              логи и артефакты обучения (не в git)
docs/              документация (см. docs/ARCHITECTURE.md)
```

Обзор архитектуры — [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Окружение (uv)

Python ≥3.10. Зависимости и виртуальное окружение — через [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install torch scikit-learn pyyaml pillow opencv-python matplotlib pandas
```

## Данные

Наборы не хранятся в git. Каждый поддатасет самоописывается своим `manifest.yaml`;
добавить новый набор — положить папку с `data/`, `masks/` и манифестом.

```
datasets/datasets_list/
  dataset_kaggle/   manifest.yaml + data/*.mat + masks/*.png
  dataset_tpu/      manifest.yaml + data/*.mat + masks/*.png
```

Kaggle-набор: <https://www.kaggle.com/datasets/ziangwei/irt-pvc-depth>

## Запуск

```python
from datasets import TermoDataset

ds = TermoDataset(
    root_dir="datasets/datasets_list",
    include=["dataset_tpu"],   # None — все поддатасеты
    transform=None,            # transform(data, mask) -> (data, mask)
)

data, mask = ds[0]             # (C, 256, 256), (256, 256)
```

Пример с аугментациями и экспортом видео/маски — [datasets/example.py](datasets/example.py).
