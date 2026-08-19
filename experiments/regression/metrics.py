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


class RegressionMetrics:
    """Метрики регрессии глубины (мм). Тот же интерфейс, что у `Metrics`.

    Основные (регрессионные): mae_mm, rmse_mm, r2, bias_mm, medae_mm.
    Дополнительные (для сравнимости с классификацией): предсказание квантуется
    к ближайшему узлу kaggle-глубин (`depth_bins_mm`) → accuracy / QWK. Осмысленны
    в первую очередь на kaggle-сетке; для непрерывных tpu-глубин это грубая оценка.
    """

    def __init__(self, y_true_mm, y_pred_mm, domain=None, bins_mm=None):
        self.y_true = np.asarray(y_true_mm, dtype=float)
        self.y_pred = np.asarray(y_pred_mm, dtype=float)
        self.domain = None if domain is None else np.asarray(domain)
        if bins_mm is None:
            bins_mm = getattr(CFG.classes, "depth_bins_mm", None)
        self.bins = (None if bins_mm is None or len(bins_mm) == 0
                     else np.asarray(sorted(bins_mm), dtype=float))

    @classmethod
    def from_loader(cls, model, loader, device, target_mean, target_std,
                    domain=None, **kwargs):
        """Прогон модели по loader. Денормировка выхода: pred_mm = out*std+mean.

        Таргет в loader тоже нормирован — денормируем так же. `domain` (опц.) —
        массив меток домена той же длины, что и датасет loader'а (порядок совпадает
        при shuffle=False), для раздельных метрик по доменам.
        """
        import torch
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                out = model(x.to(device)).cpu().numpy().reshape(-1)
                yp.append(out * target_std + target_mean)
                yt.append(y.numpy().reshape(-1) * target_std + target_mean)
        return cls(np.concatenate(yt), np.concatenate(yp),
                   domain=domain, **kwargs)

    def _quantize(self, v):
        """Ближайший узел kaggle-глубин (мм) → индекс класса."""
        return np.abs(v[:, None] - self.bins[None, :]).argmin(axis=1)

    def compute(self) -> dict:
        e = self.y_pred - self.y_true
        ss_res = float(np.sum(e ** 2))
        ss_tot = float(np.sum((self.y_true - self.y_true.mean()) ** 2)) + 1e-12
        out = {
            "mae_mm": float(np.mean(np.abs(e))),
            "rmse_mm": float(np.sqrt(np.mean(e ** 2))),
            "medae_mm": float(np.median(np.abs(e))),
            "bias_mm": float(np.mean(e)),
            "r2": float(1.0 - ss_res / ss_tot),
        }
        if self.bins is not None and len(self.bins) > 1:
            qt, qp = self._quantize(self.y_true), self._quantize(self.y_pred)
            out["acc_quant"] = float(accuracy_score(qt, qp))
            out["qwk_quant"] = float(cohen_kappa_score(
                qt, qp, labels=list(range(len(self.bins))), weights="quadratic"))
        return out

    def by_domain(self) -> dict:
        """Метрики раздельно по каждому домену (kaggle/tpu)."""
        if self.domain is None:
            return {"all": self.compute()}
        res = {}
        for d in sorted(set(self.domain.tolist())):
            m = self.domain == d
            res[d] = RegressionMetrics(self.y_true[m], self.y_pred[m],
                                       bins_mm=self.bins).compute()
        res["all"] = self.compute()
        return res

    def summary(self) -> dict:
        if self.domain is not None:
            d = self.by_domain()
            for dom, md in d.items():
                print(f"[{dom:6}] " +
                      " | ".join(f"{k}={v:.4f}" for k, v in md.items()))
            return d
        d = self.compute()
        print(" | ".join(f"{k}={v:.4f}" for k, v in d.items()))
        return d
