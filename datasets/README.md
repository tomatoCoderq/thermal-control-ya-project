# TermoDataset

`TermoDataset` — класс `torch.utils.data.Dataset` для загрузки термо-датасета,
объединяющего несколько поддатасетов из одной корневой директории. Каждый
поддатасет описывается собственным `manifest.yaml` и содержит `.mat`-файлы
с данными и файлы масок.

## Структура данных на диске

```
root_dir/
  dataset_1/
    manifest.yaml
    <data.path>/*.mat
    <masks.path>/*<masks.file_pattern>
  dataset_2/
    manifest.yaml
    ...
```

Каждый `manifest.yaml` парсится в объект `DatasetConfig` (см. `config.py`) и
должен содержать поля `data.path`, `data.file_pattern`, `data.mat_key`,
`masks.path`, `masks.file_pattern`, `crop` (`x0`, `x1`, `y0`, `y1`).

## Конструктор

```python
TermoDataset(
    root_dir: str,
    include: Optional[list[str]] = None,
    transform: Optional[callable] = None,
)
```

| Параметр    | Тип                      | Описание                                                                 |
|-------------|---------------------------|---------------------------------------------------------------------------|
| `root_dir`  | `str`                     | Путь к папке, содержащей подпапки поддатасетов.                          |
| `include`   | `list[str] \| None`       | Имена подпапок, которые нужно включить. `None`/пусто — берутся все.       |
| `transform` | `callable \| None`        | Функция аугментации вида `transform(data, mask) -> (data, mask)`.        |

### Что происходит при инициализации

1. Перечисляются все подпапки `root_dir` (только директории).
2. Если задан `include`, список подпапок фильтруется по имени (`os.path.basename`).
3. Для каждой оставшейся подпапки читается `manifest.yaml` → `DatasetConfig`.
4. Ищутся все файлы `data.path/*` с расширением `data.file_pattern`.
5. Для каждого `.mat`-файла формируется путь к соответствующей маске:
   `masks.path/<stem><masks.file_pattern>`, где `<stem>` — имя `.mat`-файла без расширения.
6. Тройки `(mat_path, mask_path, config)` сохраняются в `self.items`.

## `__len__`

```python
def __len__(self) -> int:
    return len(self.items)
```

Возвращает общее количество собранных сэмплов по всем включённым поддатасетам.

## `__getitem__`

```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]
```

## Формат `manifest.yaml`

Каждая подпапка поддатасета должна содержать файл `manifest.yaml` следующего вида:

```yaml
name: TPUdataset

data:
  path: data
  file_pattern: ".mat"
  mat_key: "data"
  dtype: float32

masks:
  path: masks
  file_pattern: ".png"

crop:
  x0: null
  x1: null
  y0: null
  y1: null
```

| Поле                  | Описание                                                                 |
|-----------------------|---------------------------------------------------------------------------|
| `name`                | Имя датасета.                                                             |
| `data.path`           | Путь к папке с `.mat`-файлами, относительно папки датасета.               |
| `data.file_pattern`   | Расширение файлов данных (например, `.mat`), проверяется через `endswith`.|
| `data.mat_key`        | Имя переменной внутри `.mat`-файла, где лежит массив данных.              |
| `data.dtype`          | Тип, к которому приводятся данные (например, `float32`).                  |
| `masks.path`          | Путь к папке с масками, относительно папки датасета.                      |
| `masks.file_pattern`  | Расширение файлов масок (например, `.png`), также используется как суффикс при формировании имени файла маски по `stem` `.mat`-файла. |
| `crop.x0/x1/y0/y1`    | Координаты обрезки кадра. `null` — координата не задана. Кроп применяется только если заданы **все четыре** координаты; если хотя бы одна `null` — кроп полностью пропускается. |

## Пример использования

```python
from termo_dataset import TermoDataset

ds = TermoDataset(
    root_dir="datasets_list",
    include=["dataset_tpu"],   # или None, чтобы взять все поддатасеты
    transform=my_augmentations,
)

print(len(ds))
data, mask = ds[0]
print(data.shape, mask.shape)   # (C, 256, 256) (256, 256)
```

## Известные ограничения

- Кроп применяется только при полном наборе координат (все четыре не `None`).
- Срез `:1500` по каналам в `_apply_crop` применяется только когда кроп
  активен; без заданного `crop` данные проходят с полным числом каналов —
  это несогласованное поведение между "кроп есть" и "кропа нет".
- Нормализация — глобальная (одно `mean`/`std` на весь тензор), а не по
  каждому каналу отдельно.