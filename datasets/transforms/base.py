"""Interface for data transforms. Works for both train and inference."""

import random

import numpy as np


class Transform:
    """Doesn't change channels by default"""

    def out_channels(self, c_in: int) -> int:
        return c_in


class Compose(Transform):
    """Sequentially applies transforms"""

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, data, mask=None):
        for t in self.transforms:
            data, mask = t(data, mask)
        return data, mask

    def out_channels(self, c_in: int = 0) -> int:
        """
        c_in channels is initial number of channels
        Returns final number after all transforms applied.
        """
        c = c_in
        for t in self.transforms:
            c = t.out_channels(c)
        return c


class RandomChoice(Transform):
    """Applies k random transforms from the list (augmentation only)."""

    def __init__(self, transforms: list[Transform], k: int = 2):
        self.transforms = transforms
        self.k = k

    def __call__(self, data, mask=None):
        for t in random.sample(self.transforms, self.k):
            data, mask = t(data, mask)
        return data, mask


class Stack(Transform):
    """Stacks channels into final (C, H, W) tensor"""

    def __init__(self, extractors: list[Transform]):
        self.extractors = extractors

    def __call__(self, data, mask=None):
        chans = np.concatenate([e(data, mask)[0] for e in self.extractors], axis=0)
        return chans, mask

    def out_channels(self, c_in: int = 0) -> int:
        '''Rewritten as Stack changes channels, not default behavior'''
        return sum(e.out_channels(c_in) for e in self.extractors)
