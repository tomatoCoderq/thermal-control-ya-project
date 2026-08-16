# SegFormer

Ветка `segformer` основана на ветке `augmentation`. В ней нет классификации
глубины и отдельной копии аугментации.

Общий Dataset возвращает квадратные PPT-тензоры `[3,256,256]`. SegFormer MIT-B0
принимает их как трёхканальные изображения и предсказывает два класса:
`background` и `defect`.

Обучение использует Cross Entropy + Dice loss. Разбиение выполняется по
`video_id` до аугментации. Сохраняются Dice, IoU, precision, recall и pixel
accuracy.

## Kaggle

```bash
git clone -b segformer \
  https://github.com/tomatoCoderq/thermal-control-ya-project.git
cd thermal-control-ya-project
pip install -q -r requirements-segformer.txt

python -m irt_data.cache \
  --sources /kaggle/input/irt-pvc-depth/data \
  --out /kaggle/working/thermal/cache

python train_segformer.py \
  --config configs/segformer.yaml \
  --data-root /kaggle/input/irt-pvc-depth/data \
  --mask-root /kaggle/input/irt-pvc-depth/labels/manual_mask \
  --work-dir /kaggle/working/thermal
```

Точные Input-пути нужно скопировать из панели Kaggle. Для первого скачивания
`nvidia/mit-b0` требуется включить Internet. Лучший checkpoint сохраняется как
`segformer/best.pt`, использованный split — `segformer/split.json`.

Параметры входа и аугментации меняются только в
`configs/augmentation_ppt.yaml`; параметры модели — в `configs/segformer.yaml`.
