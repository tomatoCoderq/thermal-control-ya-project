# Данные и запуск

## Куда класть датасеты

```
datasets/
├── dataset_kaggle/            # обучение + валидация (есть маски)
│   ├── data/                  # R_*.mat, Z_*.mat   (38 записей)
│   └── labels/
│       ├── automated_mask/    # *.png  (используется по умолчанию)
│       └── manual_mask/       # *.png  (опционально)
└── dataset_tpu/               # инференс (масок нет)
    └── *.mat                  # Sample N Static.mat
```

Kaggle-набор: <https://www.kaggle.com/datasets/ziangwei/irt-pvc-depth>

Пути можно переопределить в `5th-polynom-exps/config.yaml` (`paths.kaggle_dir`, `paths.tpu_dir`).
Кэши и логи создаются сами: `features_p5/`, `features_tpu/`, `runs/`.

## Запуск

```bash
# use uv
uv init
uv venv
source .venv/bin/activate
uv sync

# зависимости
uv pip install torch timm pydantic pyyaml scipy scikit-learn pillow opencv-python matplotlib pandas

# main (обучение на kaggle + инференс на TPU)
cd 5th-polynom-exps && python main.py

# ноутбуки — запускать из папки notebooks/
5th-polynom-exps/notebooks/cnn-p5-experiment.ipynb
```

Итоги экспериментов — в `report.html` (корень проекта).
