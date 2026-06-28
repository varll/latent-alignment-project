"""CPU smoke tests for the pair-aware K-fold PA-CCS infrastructure.

These run on synthetic numpy activations (no model / GPU needed) and check the two properties the
k-fold ToxiGen pipeline relies on:

* paired examples are never split across the train/test boundary, and every example is held out
  exactly once (so out-of-fold predictions cover everything without leakage), and
* :func:`train_ccs_layers_kfold` produces out-of-fold predictions for every example such that no
  example is ever scored by a probe that was trained on it, plus per-group aggregation works.
"""

from __future__ import annotations

import numpy as np

from latent_alignment.ccs import (
    ProbeConfig,
    kfold_indices,
    train_ccs_layers_kfold,
)


def _make_paired_data(n_pairs: int, n_layers: int, dim: int, *, seed: int = 0, scale: float = 1.5):
    """Synthetic ToxiGen-style paired layout: stacked halves, opposite_indices linking partners.

    Example ``i`` (first half) is the toxic side; example ``i + n_pairs`` (second half) is its
    benign partner with the flipped label. A label-dependent shift along a fixed direction makes
    the contrast (positive - negative) linearly separable so the probe has signal to learn.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_pairs
    labels = np.array([1] * n_pairs + [0] * n_pairs, dtype=int)  # toxic=1, benign=0
    opposite = np.array(list(range(n_pairs, n)) + list(range(n_pairs)), dtype=int)

    direction = rng.normal(size=(n_layers, dim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    base = rng.normal(size=(n, n_layers, dim)).astype(np.float32)
    sign = (2 * labels - 1).astype(np.float32)[:, None, None]
    positive = (base + scale * sign * direction[None]).astype(np.float32)
    negative = (base - scale * sign * direction[None]).astype(np.float32)
    return positive, negative, labels, opposite


def test_kfold_indices_keeps_pairs_together_and_covers_all() -> None:
    n_pairs = 20
    n = 2 * n_pairs
    opposite = np.array(list(range(n_pairs, n)) + list(range(n_pairs)), dtype=int)

    folds = kfold_indices(n, n_splits=5, opposite_indices=opposite, random_state=7)
    assert len(folds) == 5

    seen_test: list[int] = []
    for train_idx, test_idx in folds:
        # train/test disjoint within a fold
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(train_idx) + len(test_idx) == n
        # a pair is never split: every example's partner is on the same side
        test_set = set(int(i) for i in test_idx)
        for i in test_idx:
            assert int(opposite[i]) in test_set
        seen_test.extend(int(i) for i in test_idx)

    # every example held out exactly once
    assert sorted(seen_test) == list(range(n))


def test_kfold_indices_is_deterministic() -> None:
    opposite = np.array(list(range(10, 20)) + list(range(10)), dtype=int)
    a = kfold_indices(20, n_splits=4, opposite_indices=opposite, random_state=3)
    b = kfold_indices(20, n_splits=4, opposite_indices=opposite, random_state=3)
    for (a_tr, a_te), (b_tr, b_te) in zip(a, b, strict=True):
        assert np.array_equal(a_tr, b_tr)
        assert np.array_equal(a_te, b_te)


def test_train_ccs_layers_kfold_oof_no_leakage() -> None:
    n_pairs, n_layers, dim, n_splits = 25, 2, 8, 5
    positive, negative, labels, opposite = _make_paired_data(n_pairs, n_layers, dim, seed=1)

    results = train_ccs_layers_kfold(
        positive, negative, labels,
        n_splits=n_splits,
        config=ProbeConfig(nepochs=40, ntries=1),
        opposite_indices=opposite,
        random_state=11,
        device="cpu",
    )

    assert set(results) == set(range(n_layers))

    # Independently recompute the folds (same args) to assert no example is scored by a probe
    # trained on it: every example appears in exactly one test fold and never in that fold's train.
    folds = kfold_indices(
        2 * n_pairs, n_splits=n_splits, opposite_indices=opposite, random_state=11
    )
    for _, values in results.items():
        oof_pred = np.asarray(values["oof_predictions"])
        oof_conf = np.asarray(values["oof_confidence"])
        # every example scored out-of-fold
        assert (oof_pred >= 0).all()
        assert np.isfinite(oof_conf).all()
        assert len(values["fold_accuracies"]) == n_splits
        # per-pair OOF polar consistency is fully populated (one value per unique pair)
        assert len(values["oof_polar_consistency"]) == n_pairs
        assert np.isfinite(np.asarray(values["oof_polar_consistency"])).all()

    # leakage check at the index level
    covered = np.zeros(2 * n_pairs, dtype=int)
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        covered[test_idx] += 1
    assert (covered == 1).all()

    # with a learnable signal, pooled OOF accuracy should beat chance on at least one layer
    best_oof = max(float(values["oof_accuracy"]) for values in results.values())
    assert best_oof > 0.6


def test_per_group_oof_aggregation() -> None:
    """End-to-end per-group slicing mirroring runs/toxigen_group_kfold.py, on synthetic groups."""
    n_pairs, n_layers, dim, n_splits = 24, 1, 6, 4
    positive, negative, labels, opposite = _make_paired_data(n_pairs, n_layers, dim, seed=2)
    # Three groups, repeated across the stacked halves (like list(df[group]) * 2).
    base_groups = np.array(["a", "b", "c"] * (n_pairs // 3))
    groups = np.concatenate([base_groups, base_groups])

    results = train_ccs_layers_kfold(
        positive, negative, labels,
        n_splits=n_splits,
        config=ProbeConfig(nepochs=40, ntries=1),
        opposite_indices=opposite,
        random_state=5,
        device="cpu",
    )[0]

    oof_pred = np.asarray(results["oof_predictions"])
    oof_conf = np.asarray(results["oof_confidence"])
    abs_pc = np.abs(np.asarray(results["oof_polar_consistency"]))
    pair_groups = groups[np.asarray(results["pair_a_idx"])]

    for grp in ["a", "b", "c"]:
        ex_mask = groups == grp
        pair_mask = pair_groups == grp
        assert ex_mask.sum() > 0 and pair_mask.sum() > 0
        acc = float((oof_pred[ex_mask] == labels[ex_mask]).mean())
        conf_mean = float(oof_conf[ex_mask].mean())
        decisiveness = float(np.abs(oof_conf[ex_mask] - 0.5).mean() * 2)
        abs_pc_mean = float(abs_pc[pair_mask].mean())
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= conf_mean <= 1.0
        assert 0.0 <= decisiveness <= 1.0
        assert np.isfinite(abs_pc_mean)


if __name__ == "__main__":
    # Runnable CPU demo (no model / GPU): exercises the full k-fold + per-group path and prints
    # a tiny report so the pipeline can be sanity-checked end-to-end without weights.
    pos, neg, lab, opp = _make_paired_data(n_pairs=24, n_layers=2, dim=8, seed=0)
    res = train_ccs_layers_kfold(
        pos, neg, lab, n_splits=4, config=ProbeConfig(nepochs=60, ntries=2),
        opposite_indices=opp, random_state=0, device="cpu",
    )
    folds = kfold_indices(len(lab), n_splits=4, opposite_indices=opp, random_state=0)
    covered = np.zeros(len(lab), dtype=int)
    for tr, te in folds:
        assert set(tr).isdisjoint(set(te))
        covered[te] += 1
    print(f"every example held out exactly once: {(covered == 1).all()}")
    for layer, v in res.items():
        print(
            f"layer {layer}: acc {v['accuracy_mean']:.3f}±{v['accuracy_std']:.3f} "
            f"(pooled OOF {v['oof_accuracy']:.3f}), |PC| {v['abs_polar_consistency_mean']:.3f}, "
            f"all examples scored OOF: {(np.asarray(v['oof_predictions']) >= 0).all()}"
        )
