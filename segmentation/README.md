# Segmentation (U-Net / Attention U-Net / Mamba-UNet)

Термическая сегментация дефектов: **yaml → `irt_data` → фичи (TSR / Fourier / PCA) → кроп 256×256 → train/eval**.

Без `.npy`-кэша каждый батч читает `.mat` через `MatIOBackend` — это очень медленно (ты это уже видела). Сначала кэш, потом train.

---

## 0. Окружение

```bash
cd /Users/user/Education/CVYandexCamp/thermal-control-ya-project
source /Users/user/Education/CVYandexCamp/venv/bin/activate   # или свой venv
export NO_ALBUMENTATIONS_UPDATE=1   # убрать SSL-варнинги albumentations
```

Корень репо = `thermal-control-ya-project`. Скрипты и yaml с путями `data/`, `labels/`, `artifacts/` считаются **от корня**.

---

## 1. Маски

### Kaggle (`R_*.mat`, `Z_*.mat`)

Уже лежат в:

```text
labels/kaggle_binary_masks/   # PNG 0/255, как в yaml
```

### TPU / `sample*.mat`

Бинарные (один цвет, как kaggle) и depth-gray:

```bash
# из корня репо (не из Attention U-Net/)
python scripts/make_tpu_masks.py
```

| Папка | Что |
|--------|-----|
| `labels/tpu_binary_masks/` | **0/255** — для сегментации (в yaml) |
| `labels/tpu_sample_masks/` | серый = глубина (мм), для регрессии |
| `datasets/dataset_tpu/labels/table_mask/` | 240×320 для thermo |

В `dataset_*.yaml` для sample указано: `masks: labels/tpu_binary_masks`.

---

## 2. Кэш кадров `.mat` → `.npy` (обязательно перед train)

Пишет float16 `(T,H,W)` + `artifacts/cache/index.json`.

**Одной командой по yaml** (и kaggle, и sample с `time_axis: 2`):

```bash
cd /Users/user/Education/CVYandexCamp/thermal-control-ya-project

python scripts/build_irt_cache.py --yaml segmentation/U-Net/dataset_tsr.yaml
```

Первый прогон может занять **десятки минут** (56 роликов). Повторный — быстрый (skip существующих).

Пересобрать с нуля:

```bash
python scripts/build_irt_cache.py --yaml segmentation/U-Net/dataset_tsr.yaml --overwrite
```

Вручную через модуль (эквивалент по частям):

```bash
# kaggle R_/Z_ (ось времени обычно auto)
python -m irt_data.cache --sources data --pattern 'R_*.mat' --out artifacts/cache
python -m irt_data.cache --sources data --pattern 'Z_*.mat' --out artifacts/cache

# sample*: время на оси 2
python -m irt_data.cache --sources data --pattern 'sample*.mat' --time-axis 2 --out artifacts/cache
```

Проверка:

```bash
ls artifacts/cache/*.npy | wc -l          # ≈ 56
test -f artifacts/cache/index.json && echo OK
```

Пока нет `index.json`, в логе будет:

`Cache index missing … using MatIOBackend (slow …)` — **не начинай длинный train**.

Кэш **фич** (TSR/PPT) строится сам при первом проходе в `artifacts/features/...` (тоже медленно один раз).

---

## 3. Train

Yaml по умолчанию: `segmentation/U-Net/dataset_tsr.yaml`  
(`samples_per_video`, пути, маски — правь **этот** файл и передавай `--yaml`).

### Attention U-Net (MPS + Adam)

```bash
cd segmentation/Attention\ U-Net

python train.py \
  --yaml ../U-Net/dataset_tsr.yaml \
  --variants tsr \
  --epochs 50 \
  --device mps
```

Fourier:

```bash
python train.py --yaml ../U-Net/dataset_fourier.yaml --variants fourier --epochs 50 --device mps
```

### Mamba-UNet

```bash
cd segmentation/Mamba-UNet

python train.py \
  --yaml ../U-Net/dataset_tsr.yaml \
  --variants tsr \
  --epochs 50 \
  --device mps
```

### Classic U-Net

```bash
cd segmentation/U-Net
python train.py --yaml dataset_tsr.yaml --variants tsr --epochs 50
```

Полезные флаги: `--batch-size 24`, `--workers 0`, `--lr 3e-4`, `--weight-decay 1e-4`, `--no-compile` (Attention).

### PCA (6 компонент)

Экстрактор уже в `irt_data` (`features.extractors: [pca]`). Вариант `pca` и yaml:

```bash
# кэш кадров тот же; фичи PCA появятся в artifacts/features/pca/
python scripts/build_irt_cache.py --yaml segmentation/U-Net/dataset_pca.yaml

cd segmentation/Attention\ U-Net
python train.py \
  --yaml ../U-Net/dataset_pca.yaml \
  --variants pca \
  --epochs 50 \
  --device mps \
  --workers 0
```

Каналов на входе: **`pca_components: 6`** (как TSR/Fourier) → модели с `in_ch=6`.

---

## 3a. Дообучение (resume)

Чекпоинты хранят только **веса модели** (не состояние Adam). `--epochs` = сколько **новых** эпох. `history.json` дописывается.

```bash
cd segmentation/Attention\ U-Net
export NO_ALBUMENTATIONS_UPDATE=1

# с лучшего чекпоинта ещё 30 эпох
python train.py \
  --yaml ../U-Net/dataset_tsr.yaml \
  --variants tsr \
  --resume best \
  --epochs 30 \
  --workers 0 \
  --no-compile

# или явный путь / last
python train.py --resume model_attunet_tsr_best.tar --epochs 30 --workers 0 --no-compile
python train.py --resume last --epochs 20 --workers 0 --no-compile
```

То же для Mamba / U-Net:

```bash
cd ../Mamba-UNet
python train.py --yaml ../U-Net/dataset_tsr.yaml --variants tsr --resume best --epochs 20

cd ../U-Net
python train.py --yaml dataset_tsr.yaml --variants tsr --resume best --epochs 20
```

`_best` обновится только если test IoU станет выше прежнего.

---

## 3b. Куда что сохраняется

Всё лежит **в папке модели**, из которой запускаешь `train.py` (не в корне репо).

### Веса (чекпоинты)

| Модель | Файлы |
|--------|--------|
| Attention U-Net | `segmentation/Attention U-Net/model_attunet_{tsr\|fourier}_best.tar` — лучший по **test IoU** |
| | `…_last.tar` — последняя эпоха |
| Mamba-UNet | `segmentation/Mamba-UNet/model_mamba_{tsr\|fourier}_best.tar` / `_last.tar` |
| U-Net | `segmentation/U-Net/model_unet_{tsr\|fourier}_best.tar` / `_last.tar` |

`best` перезаписывается, когда test IoU лучше предыдущего.

### Логи обучения (кривые train/test)

```text
segmentation/<Model>/runs/<variant>/
  history.json   # список эпох: train_loss, test_loss, train_iou, test_dice, …
  metrics.csv    # то же в CSV
```

Примеры:

- `segmentation/Attention U-Net/runs/tsr/history.json`
- `segmentation/Mamba-UNet/runs/fourier/metrics.csv`

### Графики

Из каталога `segmentation/`:

```bash
cd /Users/user/Education/CVYandexCamp/thermal-control-ya-project/segmentation

# Attention TSR
python -m common.plot_history "Attention U-Net/runs/tsr" \
  --save "Attention U-Net/runs/tsr/curves.png"

# Mamba
python -m common.plot_history "Mamba-UNet/runs/tsr" \
  --save "Mamba-UNet/runs/tsr/curves.png"

# U-Net
python -m common.plot_history "U-Net/runs/tsr" --save "U-Net/runs/tsr/curves.png"
```

Можно передать путь к `history.json` или к папке `runs/tsr`.  
Без GUI: добавь `--no-show` (только сохранит PNG).

---

## 3c. `samples_per_video` ≠ «число аугментаций»

В yaml:

- **`samples_per_video`** — сколько раз за эпоху **достаём** каждый ролик.  
  `len(dataset) ≈ N_videos × samples_per_video`.
- **`augs` + `apply: one_of`** — на сэмпл выбирается **одна** операция из списка (сейчас flip / flip / rot90 / swap), либо ничего. Типов ауг мало (~4–5) — это нормально.

**Почему не «только 5 уникальных картинок»?**  
Потому что почти всё разнообразие даёт **`crop.strategy: roi_random`**: каждый раз новое окно 256×256 внутри ROI. Даже при **нулевых** аугах 20 сэмплов одного видео — это **20 разных кропов**, а не 20 копий. Ауги лишь слегка перемешивают ориентацию поверх этого.

Схема одного `__getitem__`:

```text
видео → фичи → случайный кроп в ROI → (опционально одна ауга) → тензор
```

**Сколько ставить `samples_per_video`?**

Это про то, **как часто крутить каждое видео за эпоху**, а не про число типов ауг:

| Значение | Смысл |
|----------|--------|
| **8–15** | быстрые прогоны / отладка |
| **20–40** | разумный дефолт: за эпоху много разных окон по ROI |
| **50–100** | длиннее эпоха; имеет смысл, если ROI большой и кропы сильно отличаются |

Равенство «samples_per_video = число ауг (5)» **не нужно**: ауги дискретные и их мало, кропы — почти непрерывные. Повторы «байт-в-байт» будут редки, пока работает `roi_random`.

Если поставить `crop.strategy: center` (или один фиксированный кроп) и те же 5 ауг — тогда да, после ~5–10 вариантов пойдут почти дубликаты; в твоём yaml это не так.

---

## 4. Eval

```bash
cd segmentation/Attention\ U-Net
python eval.py          # ждёт model_attunet_tsr_best.tar + dataset_tsr.yaml

cd ../Mamba-UNet
python eval.py

cd ../U-Net
python eval.py
```

---

## 5. Посмотреть датасет

```bash
cd notebooks
# открыть irt_dataset_inspect.ipynb
# в первой ячейке YAML = …/dataset_tsr.yaml
```

---

## Типичные ошибки

| Симптом | Причина | Что делать |
|---------|---------|------------|
| Долго висит до 1-го батча, MatIOBackend | нет `artifacts/cache/index.json` | `python scripts/build_irt_cache.py …` |
| `can't open … Attention U-Net/scripts/make_tpu_masks.py` | скрипт из не той папки | запускать из **корня** репо |
| `zsh: bad pattern: [200~python` | вставка bracketed-paste | просто: `python train.py …` |
| albumentations SSL warning | update-check | `export NO_ALBUMENTATIONS_UPDATE=1` |
| Правила `samples_per_video` не действуют | правишь `dataset_tsr.yaml`, а запускаешь без `--yaml` / другой файл | явно `--yaml ../U-Net/dataset_tsr.yaml` |
| `mmap length is greater than file size` | битый/недописанный `.npy` (прервали кэш) | удали файл и перезапусти `build_irt_cache.py` (скрипт сам пересоберёт) |
| нет `sample9.mat` | в датасете его нет (есть 3–8, 10–21) | нормально; в кэш не попадёт |
| `train 1: 0%` долго / зависает | на MPS `num_workers>0` + первый `torch.compile` | перезапуск с `--workers 0 --no-compile` |

---

## Краткий чеклист перед train

1. `python scripts/make_tpu_masks.py` (если ещё нет `labels/tpu_binary_masks`)
2. `python scripts/build_irt_cache.py --yaml segmentation/U-Net/dataset_tsr.yaml`
3. `export NO_ALBUMENTATIONS_UPDATE=1`
4. `cd segmentation/Attention\ U-Net && python train.py --yaml ../U-Net/dataset_tsr.yaml --variants tsr --epochs 50 --device mps`



source /Users/user/Education/CVYandexCamp/venv/bin/activate && cd /Users/user/Education/CVYandexCamp/thermal-control-ya-project && MPLBACKEND=Agg PYTHONPATH=. python segmentation/common/plot_history.py "segmentation/Attention U-Net/runs/tsr" --save "segmentation/Attention U-Net/runs/tsr/curves.png" --no-show --title "Attention U-Net TSR" && python3 - <<'PY'
import json
from pathlib import Path
h=json.loads(Path('segmentation/Attention U-Net/runs/tsr/history.json').read_text())
best=max(h, key=lambda r: float(r['test_iou']))
last=h[-1]
print(f'epochs: {len(h)} ({h[0]["epoch"]}..{last["epoch"]})')
print(f'best test IoU {best["test_iou"]:.4f} dice {best["test_dice"]:.4f} @ ep {best["epoch"]}')
print(f'last  test IoU {last["test_iou"]:.4f} dice {last["test_dice"]:.4f} @ ep {last["epoch"]}')
print(f'last  train IoU {last["train_iou"]:.4f} / test {last["test_iou"]:.4f}')
PY
ls -lh "segmentation/Attention U-Net/runs/tsr/curves.png"
