"""thermo — общий модуль: конфиг, экстракторы фич, данные, модели, лоссы,
метрики, оптимизаторы, движок. Эксперименты (ноутбуки/results) живут на ветках."""
from .config import CFG, Config, load_config
from .pipeline import run, make_loaders, get_device
