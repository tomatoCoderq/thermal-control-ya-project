# thermal-control

Оценка глубины залегания дефектов по ИК-термографии. Из температурной динамики
остывания извлекаются признаки (TSR-полином или Фурье-фаза), по кропам дефектов
CNN предсказывает глубину — как классификацией (6 классов), так и регрессией (мм).
Работает на двух наборах (kaggle PVC и tpu) с переносом между ними. Мб увеличим кол-во датасетов

## Структура

```
thermo/          общий модуль: конфиг, экстракторы фич, данные, модели,
                 лоссы, метрики, оптимизаторы, движок (см. thermo/README.md)
irt_data/        пайплайн аугментаций (ROI-crop, resize, синхронная аугментация)
datasets/        данные (не в git / ищите сами кек лол)
scripts/         вспомогательные скрипты (препроцессинг, генерация масок)
notebooks/       ноутбуки
runs/            логи и артефакты обучения (не в git)
```

Эксперименты целиком лежат на ветках; на main — библиотека `thermo` и слой
аугментаций.

## Окружение (uv)

Python 3.12. Зависимости и виртуальное окружение — через [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install torch timm scikit-learn pyyaml pillow opencv-python matplotlib pandas
uv pip install "albumentations>=2,<3"   # только для аугментаций
```

## Данные

Наборы не хранятся в git. Разложить так:

```
datasets/
  dataset_kaggle/
    data/                       R_*.mat, Z_*.mat        (38 записей)
    labels/automated_mask/      *.png                   (маски по умолчанию)
    labels/manual_mask/         *.png                   (опционально)
  dataset_tpu/
    *.mat                       Sample N Static.mat
    labels/table_mask/          *.npy                   (маски глубины, мм)
```

Kaggle-набор: <https://www.kaggle.com/datasets/ziangwei/irt-pvc-depth>

Пути переопределяются в `thermo/config.yaml` (блок `paths`). Кэши признаков
(`features_*`) и логи (`runs/`) создаются автоматически.

## Запуск

```python
import thermo
from thermo.config import CFG
from thermo.data import build_index_kaggle, build_index_tpu, split_by_video
from thermo.features.cache import precompute

CFG.train.task = "regression"      # classification | regression
CFG.features.kind = "tsr"          # tsr | fourier

precompute("kaggle"); precompute("tpu")
idx = build_index_kaggle() + build_index_tpu()
tr, te = split_by_video(idx, n_test=3, domains=["kaggle", "tpu"])
thermo.run(tr, te)
```
