from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize as l2_normalize
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPProbe(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear2(F.relu(self.linear1(x))))


class CCS:
    def __init__(
        self,
        negative: np.ndarray,
        positive: np.ndarray,
        y_train: np.ndarray | None = None,
        *,
        nepochs: int = 1500,
        ntries: int = 10,
        lr: float = 0.015,
        batch_size: int = -1,
        device: str | torch.device | None = None,
        linear: bool = True,
        weight_decay: float = 0.01,
        var_normalize: bool = False,
        lambda_classification: float = 0.0,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.var_normalize = var_normalize
        self.negative = self._normalize(negative)
        self.positive = self._normalize(positive)
        self.y_train = y_train
        self.width = self.negative.shape[-1]
        self.lambda_classification = lambda_classification
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.linear = linear
        self.probe = self._new_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def _new_probe(self) -> nn.Module:
        probe: nn.Module
        if self.linear:
            probe = nn.Sequential(nn.Linear(self.width, 1), nn.Sigmoid())
        else:
            probe = MLPProbe(self.width)
        return probe.to(self.device).to(dtype=torch.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32) - x.astype(np.float32).mean(axis=0, keepdims=True)
        if self.var_normalize:
            x = x / (x.std(axis=0, keepdims=True) + 1e-6)
        return x

    def loss(self, p_negative: torch.Tensor, p_positive: torch.Tensor, y_true=None) -> torch.Tensor:
        informative_loss = (torch.min(p_negative, p_positive) ** 2).mean()
        consistent_loss = ((p_negative - (1 - p_positive)) ** 2).mean()
        if self.lambda_classification == 0 or y_true is None:
            return informative_loss + consistent_loss

        avg_pred = 0.5 * (p_negative + (1 - p_positive))
        if y_true.ndim == 1:
            y_true = y_true.view(-1, 1)
        return informative_loss + consistent_loss + self.lambda_classification * nn.BCELoss()(
            avg_pred, y_true
        )

    def train_once(self) -> float:
        negative = torch.tensor(self.negative, dtype=torch.float32, device=self.device)
        positive = torch.tensor(self.positive, dtype=torch.float32, device=self.device)
        y_tensor = None
        if self.lambda_classification != 0 and self.y_train is not None:
            y_tensor = torch.tensor(self.y_train, dtype=torch.float32, device=self.device)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.view(-1, 1)

        batch_size = len(negative) if self.batch_size == -1 else self.batch_size
        nbatches = max(1, int(np.ceil(len(negative) / batch_size)))
        optimizer = optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        last_loss = torch.tensor(float("inf"))

        for _ in range(self.nepochs):
            permutation = torch.randperm(len(negative), device=self.device)
            neg_epoch = negative[permutation]
            pos_epoch = positive[permutation]
            y_epoch = y_tensor[permutation] if y_tensor is not None else None

            for batch_idx in range(nbatches):
                start = batch_idx * batch_size
                stop = min(start + batch_size, len(negative))
                p_negative = self.probe(neg_epoch[start:stop])
                p_positive = self.probe(pos_epoch[start:stop])
                y_batch = y_epoch[start:stop] if y_epoch is not None else None
                last_loss = self.loss(p_negative, p_positive, y_batch)
                optimizer.zero_grad()
                last_loss.backward()
                optimizer.step()
        return float(last_loss.detach().cpu().item())

    def repeated_train(self) -> float:
        best_loss = float("inf")
        for _ in range(self.ntries):
            self.probe = self._new_probe()
            loss = self.train_once()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
        return best_loss

    def predict(self, negative: np.ndarray, positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        negative_t = torch.tensor(
            negative.astype(np.float32), dtype=torch.float32, device=self.device
        )
        positive_t = torch.tensor(
            positive.astype(np.float32), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            p_negative = self.best_probe(negative_t)
            p_positive = self.best_probe(positive_t)
        confidence = 0.5 * (p_negative + (1 - p_positive))
        predictions = (confidence.cpu().numpy() > 0.5).astype(int)[:, 0]
        return predictions, confidence.cpu().numpy()[:, 0]

    def accuracy(self, negative: np.ndarray, positive: np.ndarray, y_true: np.ndarray) -> float:
        predictions, _ = self.predict(negative, positive)
        acc = float((predictions == y_true).mean())
        return max(acc, 1 - acc)

    def silhouette(self, negative: np.ndarray, positive: np.ndarray) -> float:
        predictions, _ = self.predict(negative, positive)
        if len(np.unique(predictions)) == 1:
            return 0.0
        return float(silhouette_score(positive - negative, predictions, metric="cosine"))

    def contrastive_probas(
        self,
        a_negative: np.ndarray,
        a_positive: np.ndarray,
        not_a_negative: np.ndarray,
        not_a_positive: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def probas(x: np.ndarray) -> np.ndarray:
            x_t = torch.tensor(self._normalize(x), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                return self.best_probe(x_t).detach().cpu().numpy()

        return (
            probas(a_negative),
            probas(a_positive),
            probas(not_a_negative),
            probas(not_a_positive),
        )

    def polar_consistency(
        self,
        a_negative: np.ndarray,
        a_positive: np.ndarray,
        not_a_negative: np.ndarray,
        not_a_positive: np.ndarray,
    ) -> np.ndarray:
        p_a_neg, p_a_pos, p_not_a_neg, p_not_a_pos = self.contrastive_probas(
            a_negative, a_positive, not_a_negative, not_a_positive
        )
        return (
            0.5
            * ((p_a_pos - p_not_a_neg) ** 2 + (p_a_neg - p_not_a_pos) ** 2)
            * np.sign(p_a_pos - p_not_a_pos)
            * np.sign(p_not_a_neg - p_a_neg)
        )

    def contradiction_index(
        self,
        a_negative: np.ndarray,
        a_positive: np.ndarray,
        not_a_negative: np.ndarray,
        not_a_positive: np.ndarray,
    ) -> np.ndarray:
        p_a_neg, p_a_pos, p_not_a_neg, p_not_a_pos = self.contrastive_probas(
            a_negative, a_positive, not_a_negative, not_a_positive
        )
        return p_a_pos * p_not_a_pos + p_a_neg * p_not_a_neg

    def weights(self) -> tuple[np.ndarray, float]:
        if not self.linear:
            raise ValueError("Weights are only available for linear probes")
        layer = self.best_probe[0]
        return (
            layer.weight.detach().cpu().numpy().squeeze(),
            float(layer.bias.detach().cpu().item()),
        )


@dataclass(frozen=True)
class ProbeConfig:
    nepochs: int = 1500
    ntries: int = 10
    lr: float = 0.015
    batch_size: int = -1
    weight_decay: float = 0.01
    lambda_classification: float = 0.0
    normalizing: str = "mean"
    seed: int = 0


def train_ccs_layers(
    positive: np.ndarray,
    negative: np.ndarray,
    labels,
    train_idx,
    test_idx,
    *,
    config: ProbeConfig | None = None,
    opposite_indices=None,
    device: str | torch.device | None = None,
) -> dict[int, dict[str, object]]:
    config = config or ProbeConfig()
    set_seed(config.seed)
    labels = np.asarray(labels)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    n_layers = positive.shape[1]
    opposite_indices = None if opposite_indices is None else np.asarray(opposite_indices, dtype=int)
    results: dict[int, dict[str, object]] = {}

    for layer_idx in tqdm(range(n_layers), desc="Training CCS"):
        pos_train, pos_test = positive[train_idx, layer_idx, :], positive[test_idx, layer_idx, :]
        neg_train, neg_test = negative[train_idx, layer_idx, :], negative[test_idx, layer_idx, :]
        pos_train, pos_test, neg_train, neg_test = _normalize_split(
            pos_train, pos_test, neg_train, neg_test, config.normalizing
        )

        ccs = CCS(
            neg_train,
            pos_train,
            labels[train_idx].astype(np.float32),
            nepochs=config.nepochs,
            ntries=config.ntries,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            lambda_classification=config.lambda_classification,
            device=device,
        )
        ccs.repeated_train()

        paired_a_idx = np.array([], dtype=int)
        paired_not_a_idx = np.array([], dtype=int)
        if opposite_indices is not None:
            pairs = [
                (idx, opposite_indices[idx])
                for idx in test_idx
                if idx > opposite_indices[idx]
            ]
            if pairs:
                paired_a_idx = np.asarray([pair[0] for pair in pairs], dtype=int)
                paired_not_a_idx = np.asarray([pair[1] for pair in pairs], dtype=int)
        if len(paired_a_idx) == 0:
            pc = np.array([], dtype=np.float32)
            ci = np.array([], dtype=np.float32)
        else:
            pc = ccs.polar_consistency(
                negative[paired_a_idx, layer_idx, :],
                positive[paired_a_idx, layer_idx, :],
                negative[paired_not_a_idx, layer_idx, :],
                positive[paired_not_a_idx, layer_idx, :],
            )
            ci = ccs.contradiction_index(
                negative[paired_a_idx, layer_idx, :],
                positive[paired_a_idx, layer_idx, :],
                negative[paired_not_a_idx, layer_idx, :],
                positive[paired_not_a_idx, layer_idx, :],
            )

        weights, bias = ccs.weights()
        results[layer_idx] = {
            "accuracy": ccs.accuracy(neg_test, pos_test, labels[test_idx].astype(np.float32)),
            "silhouette": ccs.silhouette(neg_test, pos_test),
            "polar_consistency": pc,
            "contradiction_index": ci,
            "polar_consistency_mean": float(np.mean(pc)) if len(pc) else float("nan"),
            "contradiction_index_mean": float(np.mean(ci)) if len(ci) else float("nan"),
            "weights": weights,
            "bias": bias,
        }
    return results


def train_logistic_regression_layers(
    positive: np.ndarray,
    negative: np.ndarray,
    labels,
    train_idx,
    test_idx,
    *,
    random_state: int = 71,
) -> dict[int, dict[str, float]]:
    labels = np.asarray(labels)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    positive = positive - positive.mean(0)
    negative = negative - negative.mean(0)
    results: dict[int, dict[str, float]] = {}

    for layer_idx in range(positive.shape[1]):
        x_train = positive[train_idx, layer_idx, :] - negative[train_idx, layer_idx, :]
        x_test = positive[test_idx, layer_idx, :] - negative[test_idx, layer_idx, :]
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(x_train, labels[train_idx])
        pred = clf.predict(x_test)
        results[layer_idx] = {
            "accuracy": float(accuracy_score(labels[test_idx], pred)),
            "silhouette": 0.0
            if len(np.unique(pred)) == 1
            else float(silhouette_score(x_test, pred, metric="cosine")),
        }
    return results


def summarize_results(results: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for layer_idx, values in results.items():
        rows.append(
            {
                "layer": layer_idx,
                "accuracy": values.get("accuracy"),
                "silhouette": values.get("silhouette"),
                "polar_consistency_mean": values.get("polar_consistency_mean"),
                "contradiction_index_mean": values.get("contradiction_index_mean"),
                "bias": values.get("bias"),
            }
        )
    return rows


def kfold_indices(
    n_examples: int,
    *,
    n_splits: int = 5,
    opposite_indices: Sequence[int] | np.ndarray | None = None,
    random_state: int = 71,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pair-aware K-fold split over *examples*.

    Paired layouts (an example and its ``opposite_indices`` partner) must never be split
    across the train/test boundary, otherwise a probe could be trained on one polarity of a
    pair and tested on the other — a subtle leak. We therefore fold on **unique pair ids**
    (``min(i, opposite[i])``) and expand each fold back to the example level, so both halves
    of a pair always land in the same fold. With ``opposite_indices=None`` every example is
    its own group, i.e. an ordinary shuffled K-fold.

    Returns a list of ``(train_idx, test_idx)`` integer arrays, one per fold. The test sides
    together partition ``range(n_examples)`` (every example is held out in exactly one fold).
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    if opposite_indices is None:
        group_ids = np.arange(n_examples)
    else:
        opp = np.asarray(opposite_indices, dtype=int)
        if len(opp) != n_examples:
            raise ValueError("opposite_indices must have length n_examples")
        group_ids = np.minimum(np.arange(n_examples), opp)

    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    if n_splits > n_groups:
        raise ValueError(f"n_splits={n_splits} > number of unique pairs ({n_groups})")

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for _, test_group_pos in kf.split(unique_groups):
        test_mask = np.isin(group_ids, unique_groups[test_group_pos])
        folds.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return folds


def _unique_pairs(opposite_indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Unique polarity pairs as ``(a_idx, not_a_idx)`` using the ``a > opposite[a]`` convention.

    Mirrors :func:`train_ccs_layers`. Self-paired rows (``opposite[i] == i``, e.g. the single
    format) contribute no pairs, so the polar-consistency metrics stay empty for those datasets.
    """
    if opposite_indices is None:
        return np.array([], dtype=int), np.array([], dtype=int)
    pairs = [
        (i, int(opposite_indices[i]))
        for i in range(len(opposite_indices))
        if i > opposite_indices[i]
    ]
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    return (
        np.asarray([p[0] for p in pairs], dtype=int),
        np.asarray([p[1] for p in pairs], dtype=int),
    )


def train_ccs_layers_kfold(
    positive: np.ndarray,
    negative: np.ndarray,
    labels,
    *,
    n_splits: int = 5,
    config: ProbeConfig | None = None,
    opposite_indices=None,
    layers: Sequence[int] | None = None,
    random_state: int = 71,
    device: str | torch.device | None = None,
) -> dict[int, dict[str, object]]:
    """K-fold cross-validated PA-CCS, mirroring :func:`train_ccs_layers` but never testing on
    training data.

    For every layer we run :func:`kfold_indices` (pair-grouped, so a pair is never split), train
    a probe on the K-1 training folds and score the held-out fold. The per-example predictions and
    confidences and the per-pair polar-consistency / contradiction-index values are collected
    **out-of-fold** (OOF): each example/pair is scored only by the fold in which it was held out, so
    no example is ever scored by a probe that was trained on it.

    Each fold's probe has an arbitrary global polarity, so we fold orientation *per fold* using the
    fold's own test accuracy (flip when ``acc < 0.5``, mirroring ``max(acc, 1 - acc)``); after this
    ``1`` denotes the positive-label side consistently across folds, making the pooled OOF arrays
    directly sliceable (e.g. per target group) downstream.

    Returns ``{layer_idx: {...}}`` with per-fold lists, their mean/std aggregates, and the pooled
    OOF arrays (``oof_predictions``, ``oof_confidence`` over examples; ``oof_polar_consistency``,
    ``oof_contradiction_index`` over the unique pairs, aligned with ``pair_a_idx`` /
    ``pair_not_a_idx``).
    """
    config = config or ProbeConfig()
    set_seed(config.seed)
    labels = np.asarray(labels)
    n_examples = positive.shape[0]
    opposite_indices = None if opposite_indices is None else np.asarray(opposite_indices, dtype=int)

    folds = kfold_indices(
        n_examples,
        n_splits=n_splits,
        opposite_indices=opposite_indices,
        random_state=random_state,
    )

    pair_a_idx, pair_not_a_idx = _unique_pairs(opposite_indices)
    pair_pos = {int(a): k for k, a in enumerate(pair_a_idx)}

    layer_list = list(range(positive.shape[1])) if layers is None else list(layers)
    results: dict[int, dict[str, object]] = {}

    for layer_idx in tqdm(layer_list, desc="Training CCS (k-fold)"):
        oof_pred = np.full(n_examples, -1, dtype=int)
        oof_conf = np.full(n_examples, np.nan, dtype=np.float32)
        oof_pc = np.full(len(pair_a_idx), np.nan, dtype=np.float32)
        oof_ci = np.full(len(pair_a_idx), np.nan, dtype=np.float32)

        fold_acc: list[float] = []
        fold_sil: list[float] = []
        fold_pc_mean: list[float] = []
        fold_abs_pc_mean: list[float] = []
        fold_ci_mean: list[float] = []
        fold_bias: list[float] = []

        for train_idx, test_idx in folds:
            pos_train = positive[train_idx, layer_idx, :]
            pos_test = positive[test_idx, layer_idx, :]
            neg_train = negative[train_idx, layer_idx, :]
            neg_test = negative[test_idx, layer_idx, :]
            pos_train, pos_test, neg_train, neg_test = _normalize_split(
                pos_train, pos_test, neg_train, neg_test, config.normalizing
            )

            ccs = CCS(
                neg_train,
                pos_train,
                labels[train_idx].astype(np.float32),
                nepochs=config.nepochs,
                ntries=config.ntries,
                lr=config.lr,
                batch_size=config.batch_size,
                weight_decay=config.weight_decay,
                lambda_classification=config.lambda_classification,
                device=device,
            )
            ccs.repeated_train()

            preds, conf = ccs.predict(neg_test, pos_test)
            y_test = labels[test_idx].astype(int)
            acc_raw = float((preds == y_test).mean())
            # Fold orientation so 1 == positive-label side, consistently across folds.
            if acc_raw < 0.5:
                preds = 1 - preds
                conf = 1.0 - conf
            fold_acc.append(max(acc_raw, 1.0 - acc_raw))
            fold_sil.append(ccs.silhouette(neg_test, pos_test))
            _, bias = ccs.weights()
            fold_bias.append(bias)

            oof_pred[test_idx] = preds
            oof_conf[test_idx] = conf

            # Pairs fully contained in this test fold (kept together by kfold_indices).
            test_set = set(int(i) for i in test_idx)
            fold_pairs = [
                (int(a), int(b))
                for a, b in zip(pair_a_idx, pair_not_a_idx, strict=True)
                if int(a) in test_set
            ]
            if fold_pairs:
                a_idx = np.asarray([p[0] for p in fold_pairs], dtype=int)
                b_idx = np.asarray([p[1] for p in fold_pairs], dtype=int)
                pc = np.asarray(ccs.polar_consistency(
                    negative[a_idx, layer_idx, :], positive[a_idx, layer_idx, :],
                    negative[b_idx, layer_idx, :], positive[b_idx, layer_idx, :],
                )).reshape(-1)
                ci = np.asarray(ccs.contradiction_index(
                    negative[a_idx, layer_idx, :], positive[a_idx, layer_idx, :],
                    negative[b_idx, layer_idx, :], positive[b_idx, layer_idx, :],
                )).reshape(-1)
                for a, pc_val, ci_val in zip(a_idx, pc, ci, strict=True):
                    oof_pc[pair_pos[int(a)]] = pc_val
                    oof_ci[pair_pos[int(a)]] = ci_val
                fold_pc_mean.append(float(np.mean(pc)))
                fold_abs_pc_mean.append(float(np.mean(np.abs(pc))))
                fold_ci_mean.append(float(np.mean(ci)))

        oof_scored = oof_pred >= 0
        oof_accuracy = (
            float((oof_pred[oof_scored] == labels[oof_scored].astype(int)).mean())
            if oof_scored.any()
            else float("nan")
        )
        results[layer_idx] = {
            "n_folds": len(folds),
            "fold_accuracies": fold_acc,
            "fold_silhouettes": fold_sil,
            "fold_polar_consistency_means": fold_pc_mean,
            "fold_abs_polar_consistency_means": fold_abs_pc_mean,
            "fold_contradiction_index_means": fold_ci_mean,
            "accuracy_mean": _safe_mean(fold_acc),
            "accuracy_std": _safe_std(fold_acc),
            "silhouette_mean": _safe_mean(fold_sil),
            "silhouette_std": _safe_std(fold_sil),
            "polar_consistency_mean": _safe_mean(fold_pc_mean),
            "polar_consistency_std": _safe_std(fold_pc_mean),
            "abs_polar_consistency_mean": _safe_mean(fold_abs_pc_mean),
            "abs_polar_consistency_std": _safe_std(fold_abs_pc_mean),
            "contradiction_index_mean": _safe_mean(fold_ci_mean),
            "contradiction_index_std": _safe_std(fold_ci_mean),
            "bias_mean": _safe_mean(fold_bias),
            "oof_accuracy": oof_accuracy,
            "oof_predictions": oof_pred,
            "oof_confidence": oof_conf,
            "oof_polar_consistency": oof_pc,
            "oof_contradiction_index": oof_ci,
            "pair_a_idx": pair_a_idx,
            "pair_not_a_idx": pair_not_a_idx,
        }
    return results


def summarize_kfold_results(results: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    """Flatten :func:`train_ccs_layers_kfold` output into one row per layer (fold mean/std)."""
    rows = []
    for layer_idx, values in results.items():
        rows.append(
            {
                "layer": layer_idx,
                "n_folds": values.get("n_folds"),
                "accuracy_mean": values.get("accuracy_mean"),
                "accuracy_std": values.get("accuracy_std"),
                "oof_accuracy": values.get("oof_accuracy"),
                "silhouette_mean": values.get("silhouette_mean"),
                "silhouette_std": values.get("silhouette_std"),
                "polar_consistency_mean": values.get("polar_consistency_mean"),
                "polar_consistency_std": values.get("polar_consistency_std"),
                "abs_polar_consistency_mean": values.get("abs_polar_consistency_mean"),
                "abs_polar_consistency_std": values.get("abs_polar_consistency_std"),
                "contradiction_index_mean": values.get("contradiction_index_mean"),
                "contradiction_index_std": values.get("contradiction_index_std"),
                "bias_mean": values.get("bias_mean"),
            }
        )
    return rows


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if len(values) else float("nan")


def _normalize_split(pos_train, pos_test, neg_train, neg_test, normalizing: str):
    modes = _parse_normalizing(normalizing)
    for mode in modes:
        if mode == "mean":
            pos_mean = pos_train.mean(0)
            neg_mean = neg_train.mean(0)
            pos_train = pos_train - pos_mean
            pos_test = pos_test - pos_mean
            neg_train = neg_train - neg_mean
            neg_test = neg_test - neg_mean
        elif mode == "median":
            pos_median = np.median(pos_train, 0)
            neg_median = np.median(neg_train, 0)
            pos_train = pos_train - pos_median
            pos_test = pos_test - pos_median
            neg_train = neg_train - neg_median
            neg_test = neg_test - neg_median
        elif mode == "l2":
            pos_train = l2_normalize(pos_train, norm="l2", axis=1)
            pos_test = l2_normalize(pos_test, norm="l2", axis=1)
            neg_train = l2_normalize(neg_train, norm="l2", axis=1)
            neg_test = l2_normalize(neg_test, norm="l2", axis=1)
        else:
            raise ValueError(f"Unknown normalizing mode: {mode}")
    return pos_train, pos_test, neg_train, neg_test


def _parse_normalizing(normalizing: str | None) -> list[str]:
    if normalizing is None:
        return []
    modes = [mode.strip().lower() for mode in normalizing.replace("+", ",").split(",")]
    modes = [mode for mode in modes if mode and mode not in {"raw", "none"}]
    if any(mode in {"raw", "none"} for mode in modes) and len(modes) > 1:
        raise ValueError("raw/none cannot be combined with other normalizing modes")
    return modes
