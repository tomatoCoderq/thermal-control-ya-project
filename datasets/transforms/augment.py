"""Spatial augmentations (flips, rotations, transpose)"""

import random

import numpy as np

from .base import Transform

class HorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data, mask=None):
        if random.random() < self.p:
            data = np.flip(data, axis=2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()
        return data, mask


class VerticalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data, mask=None):
        if random.random() < self.p:
            data = np.flip(data, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()
        return data, mask


class Transpose(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data, mask=None):
        if random.random() < self.p:
            data = np.swapaxes(data, 1, 2).copy()
            if mask is not None:
                mask = np.swapaxes(mask, 0, 1).copy()
        return data, mask


class RandomRotate90(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data, mask=None):
        if random.random() < self.p:
            k = random.randint(1, 3)
            data = np.rot90(data, k, axes=(1, 2)).copy()
            if mask is not None:
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return data, mask
