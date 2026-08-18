# thermo — общий модуль

Единая библиотека, в которую сведены дубли из `5th-polynom-exps`,
`fourier-exps`, `regression-experiments`. На **main** живёт эта библиотека +
слой аугментаций `irt_data`; **эксперименты (ноутбуки/results) — на ветках**,
они делают `import thermo`.

## Две главные «ручки»

- `CFG.train.task` — `classification | regression` (голова/лосс/метрики/метка);
- `CFG.features.kind` — `tsr | fourier` (экстрактор признаков);
- `CFG.features.deriv` — `p5 | p5+d1 | p5+d1+d2` (доп. каналы-производные).

Число каналов на входе модели считается автоматически: `CFG.features.in_channels`.

## Структура

```
thermo/
  config.py          единый pydantic-конфиг (+ config.yaml)
  features/
    __init__.py      build_extractor(cfg) → extract(mat)->(C,H,W)
    tsr.py           полином 5° (6 коэфф.)
    fourier.py       фаза rFFT (raw|sin_cos, окно, detrend)   ← из fourier-exps
    cache.py         precompute+кэш по (kind, domain)
  data/
    index.py         build_index_kaggle/tpu, split_by_video, нормировки
    dataset.py       CropDataset(task, augment)   ← слияние cls+reg датасетов
  models.py          SmallCNN + build_model(task, in_ch)
  losses.py          cls (ce/ws/ls/cost/qwk) + reg (smooth_l1/l1/mse)
  metrics.py         Metrics + RegressionMetrics
  optimizers.py      adam | adamw | muon
  engine.py          общий train/eval (ветвление по task)
  pipeline.py        run(train_idx, test_idx) — оркестрация
```

## Пример

```python
import thermo
from thermo.config import CFG
from thermo.data import build_index_kaggle, build_index_tpu, split_by_video
from thermo.features.cache import precompute

CFG.train.task = "regression"          # или "classification"
CFG.features.kind = "fourier"          # или "tsr"
CFG.features.phase_encoding = "sin_cos"

# кэш признаков (один раз)
precompute("kaggle"); precompute("tpu")

idx = build_index_kaggle() + build_index_tpu()
tr, te = split_by_video(idx, n_test=3, domains=["kaggle", "tpu"])
out = thermo.run(tr, te)               # обучение + метрики по доменам
```

Аугментации on-the-fly: `CFG.regression.augment = True` (флип/поворот на кропе,
через `irt_data.transforms`, без экстракта фич и без диска).
