# Архитектура (main)

Задача проекта — оценка дефектов по ИК-термографии (сегментация масок / глубина
залегания). На `main` живёт **не пайплайн обучения, а слой данных**: единый
загрузчик `TermoDataset`, который сводит разнородные термо-датасеты к одному
формату `(data, mask)`. Обучение и эксперименты — в ноутбуках и на ветках.

Прежний унифицированный пайплайн `thermo/` (config+features+engine+pipeline)
переведён в `thermo_deprecated/` и больше не развивается — см.
[thermo_deprecated/README.md](../thermo_deprecated/README.md).

## Структура

```
datasets/              слой данных (главное на main)
  datasets.py          TermoDataset — torch.utils.data.Dataset
  config.py            DatasetConfig/DataConfig/MaskConfig/CropConfig из manifest.yaml
  example.py           пример: аугментации + сборка видео/маски
  README.md            подробное описание TermoDataset
  datasets_list/       рабочие поддатасеты (gitignored)
    dataset_kaggle/    manifest.yaml + data/*.mat + masks/*.png
    dataset_tpu/       manifest.yaml + data/*.mat + masks/*.png
  dataset_defects/     сырьё CFRP/GFRP: последовательности кадров *.csv (gitignored)
  dataset_for_segmentation/  CFRP-набор с аннотированными масками сегментации

scripts/               препроцессинг
  make_tpu_masks.py    генерация масок глубины для dataset_tpu из таблицы
  thermal_to_tiff.py   экспорт кадра .mat в 32-битный float TIFF (реальные °C)
  thermal_to_video.py  .mat → colormap-видео (mp4/gif)

experiments/notebooks/ эксперименты (ссылаются на off-main код / ветки)
thermo_deprecated/     прежний пайплайн thermo (не используется)
runs/                  логи и артефакты обучения (gitignored)
docs/                  документация, статьи, отчёты
```

## Слой данных: TermoDataset

Один загрузчик поверх нескольких поддатасетов из общей корневой директории.
Каждый поддатасет самоописывается своим `manifest.yaml`, поэтому добавить новый
набор — это положить папку с `data/`, `masks/` и манифестом, код менять не надо.

```
root_dir/                       (например datasets/datasets_list)
  <subdataset>/
    manifest.yaml               DataConfig + MaskConfig + CropConfig
    <data.path>/*.mat           массив под ключом data.mat_key
    <masks.path>/<stem><ext>    маска, имя = имя .mat без расширения
```

`TermoDataset(root_dir, include=None, transform=None, standard_size=(256,256))`:

- `include` — список имён поддатасетов; `None` → берутся все;
- `transform(data, mask) -> (data, mask)` — синхронная аугментация до тензоров;
- `standard_size` — общий размер, к которому приводятся все сэмплы.

### Поток одного сэмпла (`__getitem__`)

```
loadmat(mat)[mat_key]              (H, W, C) float32
mask = Image.open(mask_path)       (H, W)
_apply_crop(data, mask, crop)      кроп только если заданы все x0,x1,y0,y1
transform(data, mask)              опциональная аугментация (numpy)
→ torch tensor, permute → (C, H, W)
F.interpolate → standard_size      data: bilinear, mask: nearest
(data - mean) / (std + 1e-8)       глобальная нормировка по всему тензору
return data, mask
```

### Формат manifest.yaml

```yaml
name: TPUdataset
data:  { path: data,  file_pattern: ".mat", mat_key: "data", dtype: float32 }
masks: { path: masks, file_pattern: ".png" }
crop:  { x0: null, x1: null, y0: null, y1: null }   # все four заданы → кроп
```

## Особенности и ограничения

- Кроп применяется только при всех четырёх координатах; иначе полностью
  пропускается. При активном кропе к каналам добавляется срез `:1500` —
  поведение с кропом и без него несогласовано.
- Нормировка глобальная (одно mean/std на тензор), не поканальная.
- Данные (`data/`, `labels/`, `*.mat`, `*.png`, `*.zip`, `dataset_defects/`,
  `dataset_for_segmentation/`) и `runs/` — в `.gitignore`; в репозитории только код.

## Соответствие веткам

`main` — слой данных. Пайплайны обучения и эксперименты лежат по веткам
(regression_experiments, fourier-classification, self-supervised и др.);
на `main` они не запускаются, поскольку ссылаются на off-main пакеты.
