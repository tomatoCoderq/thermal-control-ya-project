import random
import cv2
import numpy as np
from datasets import TermoDataset

ROOT_DIR = "datasets_list"


class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data, mask):
        if random.random() < self.p:
            data = np.flip(data, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        return data, mask


class VerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data, mask):
        if random.random() < self.p:
            data = np.flip(data, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return data, mask


class Transpose:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data, mask):
        if random.random() < self.p:
            data = np.swapaxes(data, 0, 1).copy()
            mask = np.swapaxes(mask, 0, 1).copy()
        return data, mask


class RandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data, mask):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])   # 90 / 180 / 270
            data = np.rot90(data, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return data, mask


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data, mask):
        for t in self.transforms:
            data, mask = t(data, mask)
        return data, mask


transform = Compose([HorizontalFlip(0.5),VerticalFlip(0.5),Transpose(0.5),RandomRotate90(0.5)])
