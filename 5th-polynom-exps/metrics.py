from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)

try:
    from config import CFG
    _DEF_N = CFG.classes.n_classes
    _DEF_NAMES = list(CFG.classes.class_names)
except Exception:
    _DEF_N, _DEF_NAMES = None, None


class Metrics:
    def __init__(self, y_true, y_pred, n_classes=None, class_names=None, y_prob=None):
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_pred = np.asarray(y_pred, dtype=int)
        self.y_prob = None if y_prob is None else np.asarray(y_prob, dtype=float)
        if n_classes is None:
            n_classes = _DEF_N or int(max(self.y_true.max(), self.y_pred.max()) + 1)
        self.n_classes = int(n_classes)
        self.labels = list(range(self.n_classes))
        self.class_names = (class_names or _DEF_NAMES
                            or [str(i) for i in self.labels])

    @classmethod
    def from_loader(cls, model, loader, device, **kwargs):
        import torch
        model.eval()
        yt, yp, probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                logits = model(x.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                yp += logits.argmax(dim=1).cpu().tolist()
                yt += list(y.tolist())
        return cls(yt, yp, y_prob=np.concatenate(probs), **kwargs)

    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred, labels=self.labels)

    def per_class(self) -> dict:
        p, r, f, s = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=self.labels, zero_division=0)
        return {self.class_names[i]: dict(precision=float(p[i]), recall=float(r[i]),
                                          f1=float(f[i]), support=int(s[i]))
                for i in self.labels}

    def compute(self) -> dict:
        yt, yp = self.y_true, self.y_pred
        return {
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, labels=self.labels,
                                       average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, labels=self.labels,
                                          average="weighted", zero_division=0)),
            "ordinal_mae": float(np.mean(np.abs(yt - yp))),
            "qwk": float(cohen_kappa_score(yt, yp, labels=self.labels,
                                           weights="quadratic")),
        }

    def report(self) -> str:
        return classification_report(self.y_true, self.y_pred, labels=self.labels,
                                     target_names=self.class_names, zero_division=0)

    def to_dict(self) -> dict:
        d = self.compute()
        d["per_class"] = self.per_class()
        d["confusion"] = self.confusion().tolist()
        return d

    def summary(self) -> dict:
        d = self.compute()
        print(" | ".join(f"{k}={v:.4f}" for k, v in d.items()))
        print(self.report())
        return d
