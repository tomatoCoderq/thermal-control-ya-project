# Документация `irt_data`

**Где лежит:** [`dataset.md`](dataset.md) (этот файл) в корне `HeatControl/`.

Пакет: [`irt_data/`](irt_data/) — config-driven PyTorch Dataset / DataLoader для семантической сегментации по данным активного теплового контроля (IRT NDT).

Связанные файлы:

| Файл | Зачем |
|------|--------|
| [`dataset.md`](dataset.md) | эта документация |
| [`dataset_demo.ipynb`](dataset_demo.ipynb) | демо обоих режимов |
| [`augmentation_example.ipynb`](augmentation_example.ipynb) | ROI + аугментации на Kaggle |
| [`configs/features_unet.yaml`](configs/features_unet.yaml) | пример features |
| [`configs/temporal_convlstm.yaml`](configs/temporal_convlstm.yaml) | пример temporal |
| [`configs/augmentation_example.yaml`](configs/augmentation_example.yaml) | ROI на все 38 Kaggle + augs |

---

## 1. Зачем это нужно

Видео термографии — тысячи кадров `(T, H, W)`. Модели бывают двух типов:

| Режим | Модели | Что отдаёт Dataset |
|-------|--------|-------------------|
| **`features`** | U-Net, обычный SegFormer | одно многоканальное изображение `[C, H, W]` (время «схлопнуто» в каналы через TSR/PCA) |
| **`temporal`** | ConvLSTM, Temporal SegFormer, TimeSformer | клип `[T, C, H, W]` (сэмпл кадров из длинного ролика) |

Один и тот же Dataset переключается режимом в YAML: `mode: features` или `mode: temporal`.

---

## 2. Данные

### 2.1 Kaggle (`archive/`)

- Видео: `archive/data/*.mat` — ключ `imageArray`, форма **`(H, W, T) = (256, 320, 1810)`**, `uint16`, `Fs ≈ 10`.
- Маски: `archive/labels/manual_mask/*.png` — пиксели `0, 51, 102, 153, 204, 255` = фон + 5 глубин.
- Классы: `pixel // 51` → `0..5`.

### 2.2 Свои Sample (`data/`)

- `data/sample*.mat` — ключ `data`, форма **`(240, 320, 2000)`**, `float32`.
- Масок сегментации пока нет → в конфиге `mask.missing: zeros` (или не считать mask-loss).
- `FPS` в файле часто `0.0` (битый) — не опираться.

### 2.3 Важно про оси

В `.mat` лежит `(H, W, T)`. При кэшировании / чтении пакет приводит к **`(T, H, W)`**.

Размеры кадров разные (256×320 vs 240×320): всегда задавайте общий `crop.size`, иначе нельзя смешивать источники в одном батче.

### 2.4 Идентификаторы видео

`video_id` = stem файла, пробелы → `_`:

- `R_002.mat` → `R_002`
- `sample3.mat` → `sample3`
- `Calib Sample 1 Static.mat` → `Calib_Sample_1_Static`

Имена в `files_meta` должны совпадать с этими id.

---

## 3. Быстрый старт

```bash
cd HeatControl
source ../venv/bin/activate   # или ваш venv

# 1) Один раз: .mat → float16 memmap (нужен для temporal и ускоряет features)
python -m irt_data.cache --sources archive/data --out artifacts/cache

# свои семплы (опционально)
python -m irt_data.cache --sources data --out artifacts/cache
```

```python
from irt_data.config import DatasetConfig
from irt_data.loaders import build_dataloader

cfg = DatasetConfig.from_yaml("configs/features_unet.yaml")
loader = build_dataloader(cfg)
batch = next(iter(loader))
print(batch["image"].shape)  # [B, C, H, W]
print(batch["mask"].shape)   # [B, H, W]
```

```python
cfg = DatasetConfig.from_yaml("configs/temporal_convlstm.yaml")
loader = build_dataloader(cfg)
batch = next(iter(loader))
print(batch["image"].shape)  # [B, T, C, H, W]
```

Без YAML:

```python
from irt_data.config import DatasetConfig, SourceConfig, TemporalConfig, CropConfig

cfg = DatasetConfig(
    mode="temporal",
    sources=[SourceConfig(root="archive/data", masks="archive/labels/manual_mask")],
    cache_dir="artifacts/cache",
    temporal=TemporalConfig(sampler="window", num_frames=16, frame_range=(0, 400)),
    crop=CropConfig(size=(256, 256), strategy="random"),
    train=True,
)
loader = build_dataloader(cfg)
```

---

## 4. Архитектура (поток данных)

```text
.mat
  └─► cache.py  →  artifacts/cache/<id>.npy  float16 (T,H,W) + index.json
                        │
                        ▼
              NpyMemmapBackend.read_frames / read_all
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   mode=features                mode=temporal
   FeatureExtractor             FrameSampler
   (TSR / PCA → H,W,C)          (uniform/window/keypoints → indices)
          │                           │
          └─────────────┬─────────────┘
                        ▼
                  RoiCropper  (один CropBox на весь клип + маску)
                        ▼
              TransformPipeline (albumentations)
              image/images + mask вместе
                        ▼
                TensorFormatter
                        ▼
              dict → DataLoader / collate
```

**Dataset только оркестрирует.** Математика и I/O живут в отдельных модулях.

---

## 5. Структура пакета

```text
irt_data/
  __init__.py          # lazy exports: DatasetConfig, IRTDataset, build_dataloader
  __main__.py          # python -m irt_data → cache
  config.py            # dataclasses + YAML
  cache.py             # .mat → .npy float16
  io_backend.py        # memmap / mat / masks
  features.py          # TSR, PCA, composite, disk-cache признаков
  samplers.py          # выбор кадров + jitter/drop
  crops.py             # ROI / random / center crop + pad
  transforms.py        # albumentations pipeline
  formatter.py         # normalize + mask → классы → torch
  dataset.py           # IRTDataset
  loaders.py           # DataLoader, stack/pad collate
```

---

## 6. Кэш (`python -m irt_data.cache`)

### Зачем

Одно видео ≈ 600 МБ float32. `scipy.io.loadmat` читает файл целиком. Для случайных 16–30 кадров это дорого. Кэш:

- `artifacts/cache/<video_id>.npy` — `float16`, layout `(T, H, W)`
- `artifacts/cache/index.json` — T/H/W, min/max/mean/std, путь к источнику, fps

`NpyMemmapBackend` делает `np.load(..., mmap_mode="r")[indices]` — с диска поднимаются только нужные кадры.

### CLI

```bash
python -m irt_data.cache --sources archive/data --out artifacts/cache
python -m irt_data.cache --sources data archive/data --overwrite
python -m irt_data.cache --sources archive/data/R_002.mat --out artifacts/cache
```

| Флаг | Смысл |
|------|--------|
| `--sources` | папки и/или отдельные `.mat` |
| `--out` | каталог кэша (по умолчанию `artifacts/cache`) |
| `--pattern` | glob внутри папки (default `*.mat`) |
| `--time-axis` | принудительная ось времени (иначе auto) |
| `--overwrite` | пересобрать существующие `.npy` |

Если `index.json` нет, Dataset падает в медленный `MatIOBackend` (LRU на 1 видео) и пишет warning.

---

## 7. Конфиг (YAML / dataclasses)

Загрузка: `DatasetConfig.from_yaml(path)` / `from_dict(dict)` / сохранение `to_yaml(path)`.

### 7.1 Корневые поля `DatasetConfig`

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| `mode` | `features` \| `temporal` | `features` | режим Dataset |
| `sources` | list | `[]` | откуда брать `.mat` и маски |
| `cache_dir` | str | `artifacts/cache` | memmap-кэш |
| `files_meta` | dict[id → FileMeta] | `{}` | ROI, frame_range, keypoints |
| `features` | FeatureConfig | | только для `mode: features` |
| `temporal` | TemporalConfig | | только для `mode: temporal` |
| `crop` | CropConfig | | кроп / ROI |
| `augs` | AugConfig | | albumentations |
| `norm` | NormConfig | | нормализация |
| `mask` | MaskConfig | | кодирование маски |
| `loader` | LoaderConfig | | DataLoader |
| `samples_per_video` | int | `1` | сколько раз видео в epoch (разный кроп/окно) |
| `seed` | int | `0` | база для RNG |
| `train` | bool | `true` | jitter/drop и shuffle |

### 7.2 `sources[]` — `SourceConfig`

```yaml
sources:
  - root: archive/data
    masks: archive/labels/manual_mask
    pattern: "*.mat"
    time_axis: null   # или 2, если нужно форсировать (H,W,T)
```

### 7.3 `files_meta.<video_id>` — `FileMeta`

```yaml
files_meta:
  R_002:
    frame_range: [0, 200]   # интересные кадры [start, end)
    heat_start: 5
    cool_start: 50
    peak_contrast: 11
    rois:
      - {x: 85, y: 107, w: 153, h: 94}   # обычно union всех дефектов
      - {x: 160, y: 115, w: 20, h: 20}   # отдельные компоненты
    notes: ""
```

| Поле | Смысл |
|------|--------|
| `frame_range` | окно интересной части `[start, end)` |
| `rois` | список bbox `(x, y, w, h)` в пикселях |
| `heat_start` / `cool_start` / `peak_contrast` | для sampler `keypoints` |
| `fps` | опционально |
| `notes` | свободный текст |

**Приоритет окна времени:**  
`files_meta.<id>.frame_range` → `temporal.frame_range` → всё видео `[0, T)`.

### 7.4 `features` — `FeatureConfig`

```yaml
features:
  extractors: [tsr, pca]
  frame_step: 8
  max_frames: null
  poly_degree: 5
  pca_components: 3
  thermal_diff: true
  cache_dir: artifacts/features
```

| Extractor | Каналы | Что считает |
|-----------|--------|-------------|
| `tsr` | 5 | max ΔT, time-to-peak, d1@peak, mean d1/d2 на «хвосте» (log-poly fit) |
| `pca` | `pca_components` | PCT: score-карты первых PC |

Итог `C = сумма каналов` (например tsr+pca3 → **8**). Результат кэшируется в `artifacts/features/<id>__<hash>.npy`.

### 7.5 `temporal` — `TemporalConfig`

```yaml
temporal:
  sampler: window          # uniform | window | keypoints
  num_frames: 16
  stride: 2
  window_size: 16          # для window; по умолчанию = num_frames
  jitter: 1                # ±кадры (только train)
  frame_drop_p: 0.0        # случайный drop + pad (только train)
  time_pad: repeat_last    # repeat_last | reflect
  add_dt_channel: false    # второй канал dT/dt
  frame_range: [0, 400]    # глобальное окно [start, end)
```

#### Сэмплеры

| Имя | Поведение |
|-----|-----------|
| `uniform` | равномерно `num_frames` точек в `frame_range` |
| `window` | случайное окно из `window_size` кадров с шагом `stride`, затем подгонка к `num_frames` |
| `keypoints` | вокруг `heat_start` / `cool_start` / `peak_contrast` (если нет — early/mid/late) |

Если видео короче запрошенного `T` → паддинг индексов (`repeat_last` или `reflect`).

### 7.6 `crop` — `CropConfig`

```yaml
crop:
  size: [256, 256]      # [H, W]
  strategy: roi_random  # random | center | roi_random | roi_center | full
  pad_mode: reflect     # reflect | zeros  (для изображения; маска всегда zeros)
  roi_padding: 16       # расширение ROI перед кропом
```

| strategy | Поведение |
|----------|-----------|
| `random` | случайный кроп по кадру |
| `center` | центр |
| `roi_random` | случайный ROI из `files_meta`, кроп пересекает/около него |
| `roi_center` | центр первого ROI |
| `full` | весь кадр (потом всё равно режется/паддится до `size`) |

Если strategy `roi_*`, но ROI нет → fallback на `center`.  
Координаты кропа считаются **один раз** на `__getitem__` и применяются ко всем `T` кадрам и маске.

### 7.7 `augs` — `AugConfig`

```yaml
augs:
  spatial:
    - name: HorizontalFlip
      params: {p: 0.5}
    - name: Affine
      params:
        scale: [0.9, 1.1]
        rotate: [-20, 20]
        border_mode: 0
        p: 0.7
  use_replay_fallback: false
```

- `name` — имя класса **albumentations** (`HorizontalFlip`, `RandomRotate90`, `ElasticTransform`, …).
- Списки из 2 чисел в YAML → tuple (диапазоны).
- **features:** `Compose(image=..., mask=...)`.
- **temporal:** `Compose(images=[frame0..T-1], mask=...)` — одна геометрия на весь клип.
- Геометрия двигает image и mask вместе. Фотометрия (Brightness, GaussNoise) — только image (так и задумано).

Проверка: Flip/Rotate IoU=1.0; Affine/Elastic IoU≈0.999 (bilinear vs nearest).

### 7.8 `norm` — `NormConfig`

| mode | Поведение |
|------|-----------|
| `per_video` | `(x - mean) / std` из `index.json` (fallback min-max из index) |
| `per_sample` | min-max по текущему тензору |
| `none` | как есть |

### 7.9 `mask` — `MaskConfig`

```yaml
mask:
  kind: multiclass       # multiclass | binary
  num_classes: 6
  missing: zeros         # zeros | none | error
  ignore_index: 255
  # pixel_to_class: {0: 0, 51: 1, ...}  # default
```

| `kind` | Код |
|--------|-----|
| `multiclass` | `pixel // 51` → `0..5` (или кастомный `pixel_to_class`) |
| `binary` | `(mask > 0).long()` |

| `missing` | Если PNG нет |
|-----------|----------------|
| `zeros` | нулевая маска, `has_mask=False` |
| `none` | заливка `ignore_index` |
| `error` | исключение |

### 7.10 `loader` — `LoaderConfig`

```yaml
loader:
  batch_size: 4
  num_workers: 0
  shuffle: true
  pin_memory: false
  drop_last: false
  collate: stack          # stack | pad
```

---

## 8. Что возвращает `__getitem__`

```python
{
  "image": Tensor,          # features: [C,H,W]  |  temporal: [T,C,H,W]
  "mask": Tensor,           # [H,W] int64
  "video_id": str,
  "frame_indices": Tensor,  # [T] long (пусто в features)
  "crop": Tensor,           # [4] = (y0, x0, y1, x1)
  "has_mask": BoolTensor,   # скаляр
}
```

Длина Dataset = `len(video_ids) * samples_per_video`.

При битом файле: warning → video помечается bad → берётся другой индекс (до 5 попыток).

---

## 9. Формы батчей

### `collate: stack` (одинаковый T)

**features**

```text
image: [B, C, H, W]
mask:  [B, H, W]
```

**temporal**

```text
image: [B, T, C, H, W]     # C=1 или 2 при add_dt_channel
mask:  [B, H, W]
```

### `collate: pad` (разный T)

```text
image:      [B, T_max, C, H, W]
lengths:    [B]
valid_mask: [B, T_max] bool   # True = реальный кадр
mask:       [B, H, W]
```

Хвост короче `T_max` заполняется **повтором последнего кадра**.

---

## 10. Примеры конфигов

### Features (U-Net)

См. [`configs/features_unet.yaml`](configs/features_unet.yaml).

```yaml
mode: features
sources:
  - root: archive/data
    masks: archive/labels/manual_mask
features:
  extractors: [tsr, pca]
  pca_components: 3
crop:
  size: [256, 256]
  strategy: random
```

### Temporal (ConvLSTM)

См. [`configs/temporal_convlstm.yaml`](configs/temporal_convlstm.yaml).

```yaml
mode: temporal
temporal:
  sampler: window
  num_frames: 16
  frame_range: [0, 400]
```

### Аугментации + ROI на все Kaggle

См. [`configs/augmentation_example.yaml`](configs/augmentation_example.yaml) — у каждого из 38 файлов `rois` + `frame_range`, плюс список spatial-аугментаций. Визуализация: [`augmentation_example.ipynb`](augmentation_example.ipynb).

---

## 11. Как расширять

### Новая аугментация

Одна строка в YAML (`name` = класс albumentations):

```yaml
augs:
  spatial:
    - name: GridDistortion
      params: {p: 0.3}
```

### Новый feature extractor

В `features.py`:

```python
class MyFeatureExtractor:
    name = "my_feat"
    def num_channels(self) -> int: return 2
    def __call__(self, video: np.ndarray) -> np.ndarray:
        # video (T,H,W) → (H,W,C)
        ...

FEATURE_REGISTRY["my_feat"] = MyFeatureExtractor
```

```yaml
features:
  extractors: [tsr, my_feat]
```

### Новый frame sampler

В `samplers.py` — класс с `sample(T_total, rng, meta) → indices`, регистрация в `SAMPLER_REGISTRY`, затем `temporal.sampler: my_name`.

---

## 12. Edge cases

| Ситуация | Поведение |
|----------|-----------|
| Кроп выходит за кадр / ROI | `np.pad`: image — `reflect`/`zeros`, mask — всегда `zeros` |
| Видео короче `num_frames` | pad индексов: `repeat_last` / `reflect` |
| Маски нет | `mask.missing` |
| Битый `.mat` | log + skip + другой индекс |
| Нет кэша | `MatIOBackend` (медленно) + warning |
| Kaggle 256×320 + Sample 240×320 | только через общий `crop.size` |
| `roi_*` без ROI в meta | fallback на center crop |
| `samples_per_video > 1` | одно видео → несколько сэмплов с разным кропом/окном |

---

## 13. Зависимости

Уже в `CVYandexCamp/venv`:

- `torch`, `numpy`, `scipy`, `Pillow`, `pyyaml`
- `albumentations` (≥2.0, нужен `images=` для клипов)
- `scikit-learn` (PCA)
- `tqdm`, `opencv-python-headless` (ROI из компонент маски в утилитах)

Запуск из корня `HeatControl` (чтобы импорт `irt_data` и относительные пути в YAML работали).

---

## 14. Типичный пайплайн обучения (скелет)

```python
from irt_data.config import DatasetConfig
from irt_data.loaders import build_dataloader

train_cfg = DatasetConfig.from_yaml("configs/features_unet.yaml")
train_cfg.train = True
train_cfg.loader.shuffle = True

val_cfg = DatasetConfig.from_yaml("configs/features_unet.yaml")
val_cfg.train = False
val_cfg.augs.spatial = []          # без ауг на val
val_cfg.loader.shuffle = False
val_cfg.samples_per_video = 1

train_loader = build_dataloader(train_cfg)
val_loader = build_dataloader(val_cfg)

for batch in train_loader:
    x, y = batch["image"], batch["mask"]   # features: [B,C,H,W], [B,H,W]
    # logits = model(x)
    # loss = criterion(logits, y)
    ...
```

Для temporal то же самое, но `x` имеет форму `[B, T, C, H, W]` — модель должна это принимать (или `x.flatten(0,1)` / permute под ваш backbone).

---

## 15. FAQ

**Q: Почему не читать `.mat` напрямую в Dataset?**  
A: Для temporal нужен random access по кадрам. Memmap float16 ≈ в 2 раза меньше и не грузит 600 МБ целиком.

**Q: Маска крутится вместе с картинкой?**  
A: Да, для всех геометрических аугментаций из конфига. См. IoU-тест в `augmentation_example.ipynb`.

**Q: Как задать «интересную» часть ролика?**  
A: Только через уже существующий `frame_range: [start, end)` — глобально в `temporal` или per-file в `files_meta`.

**Q: Где ROI для Kaggle?**  
A: `configs/augmentation_example.yaml` → `files_meta.*.rois` (посчитаны из `manual_mask`).

**Q: Свои sample без масок — можно ли гонять Dataset?**  
A: Да, `mask.missing: zeros`, смотрите `has_mask` в батче и не считайте segmentation loss, пока нет разметки.
