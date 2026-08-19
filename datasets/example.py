import random
import cv2
import numpy as np
from datasets import TermoDataset, TermoOversampledDataset, TermoFrameDataset

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

class RandomChoice:
    def __init__(self, transforms: list, k: int = 2):
        self.transforms = transforms
        self.k = k

    def __call__(self, data, mask):
        chosen = random.sample(self.transforms, self.k)
        for t in chosen:
            data, mask = t(data, mask)
        return data, mask


transform = RandomChoice([
    HorizontalFlip(p=1.0),
    VerticalFlip(p=1.0),
    Transpose(p=1.0),
    RandomRotate90(p=1.0),
], k=3)


ds = TermoOversampledDataset(root_dir=ROOT_DIR, include=["dataset_tpu"], transform=transform, mag_coeff = 2)
print("Всего сэмплов:", len(ds))

data, mask = ds[25]
print("data:", data.shape, data.dtype)   # [2000, 256, 256]
print("mask:", mask.shape, mask.dtype)
print("mean/std:", data.mean().item(), data.std().item())

# --- видео из всех кадров data ---
video = data.numpy()   # [2000, 256, 256]

v_min, v_max = video.min(), video.max()
video_uint8 = ((video - v_min) / (v_max - v_min + 1e-8) * 255).astype(np.uint8)

out = cv2.VideoWriter(
    "example_sample.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps=30,
    frameSize=(256, 256),
    isColor=False,
)
for frame in video_uint8:
    out.write(frame)
out.release()
print("Видео сохранено в example_sample.mp4")

# --- маска отдельно, как картинка ---
mask_np = mask.numpy()
mask_uint8 = ((mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8) * 255).astype(np.uint8)
cv2.imwrite("example_mask.png", mask_uint8)
print("Маска сохранена в example_mask.png")
