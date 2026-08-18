# ConvLSTM — temporal models on raw IRT frames

Сегментация дефектов по **клипу кадров** (не TSR/Fourier-фичам). Модель накапливает
временной контекст нагрева/охлаждения через ConvLSTM на картах признаков 1/4
разрешения и декодирует маску с skip-связями от последнего кадра.

## Данные

- Kaggle: `data/R_*.mat`, `data/Z_*.mat` → `labels/kaggle_binary_masks/R_002.png` …
- TPU: `data/sample*.mat` (~2000 кадров, 240×320) → `labels/table_mask_raw/sample3.png` …
- Yaml: `dataset_convlstm.yaml` (`mode: temporal`, `num_frames: 12`)

Перед train нужен кэш кадров:

```bash
python scripts/build_irt_cache.py --yaml models/ConvLSTM/dataset_convlstm.yaml
```

## Train

```bash
cd models/ConvLSTM
export NO_ALBUMENTATIONS_UPDATE=1

python train.py --device mps --epochs 50 --workers 0 --batch-size 4
```

**Ноутбук с live-графиками:** `notebooks/convlstm_train.ipynb`

Чекпоинты: `model_convlstm_best.tar`, `model_convlstm_last.tar`  
Логи: `runs/convlstm/history.json`

## Eval

```bash
python eval.py
```

## Архитектура

| Блок | Описание |
|------|----------|
| Encoder | 3× ConvBlock, stride↓ на 1/2 и 1/4; **1 канал** на вход |
| ConvLSTM | hidden=64 @ H/4; состояние переносится по T кадрам |
| Decoder | 2× upsample + skip от последнего кадра |
| Loss | BCE + Dice, `pos_weight=10` (дефекты ~2% площади) |

Маска **статична** на весь ролик — loss считается по одному выходу после
просмотра всего клипа (финальное hidden-состояние ConvLSTM).

## Почему ConvLSTM, а не 3D-CNN / Transformer

- ConvLSTM сохраняет пространственную структуру — важно для мелких дефектов
- Для IRT-термограмм temporal dynamics (нагрев → контраст дефекта) — ключевой сигнал
- Легче и стабильнее на MPS, чем video-transformer; baseline перед Mamba/3D-CNN

Ссылки: Shi et al. ConvLSTM (2015); thermal fall-detection с BiConvLSTM + attention (2025).
