"""Classes for pipeline. Not DL/ML models"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "models" / "Thermal-Contrast"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from video_io import load_video_from_mat           # noqa: E402
from datasets.transforms import Compose            # noqa: E402


@dataclass
class Defect:
    x: int
    y: int
    depth_mm: float
    region_id: int = 0


@dataclass
class Prediction:
    mask: np.ndarray                 # (H,W) uint8 0/1
    defects: list[Defect]
    prob: np.ndarray                 # (H,W) float raw probs
    size: tuple[int, int]            # (H,W) of initial video

    def to_dict(self) -> dict:
        return {
            "size": list(self.size),
            "mask_shape": list(self.mask.shape),
            "defects": [{defect.x: defect.y} for defect in self.defects],
        }

    def save(self, output_dir: str | Path) -> None:
        """Saves mask.npy, depth.txt, meta.json to output_dir"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "mask.npy", self.mask)

        with open(out / "depth.txt", "w", encoding="utf-8") as f:
            f.write("x y depth_mm\n")
            for d in self.defects:
                depth = f"{d.depth_mm:.3f}" if d.depth_mm is not None else "nan"
                f.write(f"{d.x}\t{d.y}\t{depth}\n")
        (out / "meta.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class ThermalControlVideoPredictor:
    def __init__(self,
                 seg_model: torch.nn.Module,
                 seg_transform: Compose,
                 depth_model: torch.nn.Module | None = None,
                 depth_transform: Compose | None = None,
                 device: torch.device = torch.device("cpu"),
                 threshold: float = 0.5,
                 crop: int = 48):
        self.device = device
        self.crop = crop
        self.threshold = threshold
        self.seg_transform = seg_transform
        self.seg = seg_model.eval().to(device)
        self.depth_model = depth_model
        self.depth_transform = depth_transform
        if self.depth_model is not None:
            self.depth_model.eval().to(device)

    @torch.no_grad()
    def _segment(self, video: np.ndarray) -> dict:
        channels, _ = self.seg_transform(video) # mask not for inference
        x = torch.from_numpy(np.ascontiguousarray(
            channels)).float().unsqueeze(0)
        logits = self.seg(x.to(self.device))
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()          # (H,W)
        mask = (prob > self.threshold).astype(np.uint8)
        return {"prob": prob, "mask": mask}

    def _crop(self, mask: np.ndarray) -> list[tuple[int, int]]:
        s = self.crop
        H, W = mask.shape
        labels, n = ndimage.label(mask)
        centers: list[tuple[int, int]] = []
        for k in range(1, n + 1):
            rr, cc = ndimage.center_of_mass(labels == k)
            r0 = int(np.clip(rr - s // 2, 0, max(H - s, 0)))
            c0 = int(np.clip(cc - s // 2, 0, max(W - s, 0)))
            centers.append((r0, c0))
        return centers

    @torch.no_grad()
    def _regress(self, video: np.ndarray, centers: list[tuple[int, int]]) -> list[float]:
        if not centers:
            return []
        feats, _ = self.depth_transform(video) # (C,H,W)
        s = self.crop
        crops = [feats[:, r0:r0 + s, c0:c0 + s] for r0, c0 in centers]
        x = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).float()
        return self.depth_model(x.to(self.device)).reshape(-1).cpu().numpy().tolist()

    def predict(self, video: np.ndarray) -> Prediction:
        size = tuple(video.shape[1:]) if video.ndim == 3 else tuple(
            video.shape[:2])
        seg = self._segment(video)
        centers = self._crop(seg["mask"])
        depths = self._regress(video, centers)
        defects = [
            Defect(x=c0 + self.crop // 2, y=r0 + self.crop // 2,
                   depth_mm=h, region_id=i)
            for i, ((r0, c0), h) in enumerate(zip(centers, depths))
        ]
        return Prediction(mask=seg["mask"], defects=defects,
                          prob=seg["prob"], size=size)

    # def predict_mat(self, mat_path: str | Path, mat_key: str | None = None) -> Prediction:
    #     "из .mat напрямую (тот же ридер, что в обучении)"
    #     video = load_video_from_mat(mat_path, mat_key).numpy()
    #     return self.predict(video)


def predict_video(video: np.ndarray, predictor: ThermalControlVideoPredictor,
                  output_dir: str | Path | None = None) -> Prediction:
    "func(video, output_dir): одно предсказание + опциональный дамп на диск"
    pred = predictor.predict(video)
    if output_dir is not None:
        pred.save(output_dir)
    return pred
