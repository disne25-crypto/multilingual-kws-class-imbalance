"""Drop-in helpers for the two quick follow-up experiments (X1 and X3).

Both work on the CACHED XLS-R features, so no audio needs to be re-encoded.
Call them inside the existing loop (language -> fold -> seed). Do not change
anything else (encoder, pooling, MLP, optimizer, early stopping, metrics, or
the outputs.npz / per_class_metrics.csv / bottom20_metrics.csv files).

X1  Extreme imbalance  : extreme_train_indices(...)   -> subsample the TRAIN part only
X3  Synthetic minority : smote_features(...)          -> SMOTE in feature space (TRAIN part only)
    (optional)         : balanced_mixup(...)          -> Balanced-MixUp in feature space

Shapes: X may be (N, D) or (N, L, D) (per-layer pooled features). Both work.
"""
import numpy as np

# ---------------------------------------------------------------------------
# X1: extreme imbalance by subsampling the training split
# ---------------------------------------------------------------------------
def extreme_train_indices(train_idx, y_all, target_ir, split_seed=0, min_count=3):
    """Return a subset of `train_idx` whose class counts follow an exponential
    long tail with max/min ratio close to `target_ir`.

    - Classes are ranked by their ORIGINAL training count (largest = rank 0).
    - Target count for rank r: n_head * target_ir ** (-r / (C-1)), never more
      than the class actually has, and never fewer than `min_count`.
    - The head class keeps all its training recordings.
    - Validation and test indices are NOT touched.
    - `split_seed` must be the same for every method and training seed of a
      given language and fold, so that all methods see the SAME subset.
      Use split_seed = 1000 * fold + 7 (independent of the training seed).

    Returns (new_train_idx, info) where info has the realised counts and IR.
    """
    train_idx = np.asarray(train_idx)
    y_tr = np.asarray(y_all)[train_idx]
    classes, counts = np.unique(y_tr, return_counts=True)
    C = len(classes)
    order = np.argsort(-counts, kind="stable")          # rank 0 = most frequent
    n_head = counts[order[0]]
    rng = np.random.default_rng(split_seed)
    keep = []
    target = {}
    for r, ci in enumerate(order):
        c = classes[ci]
        want = int(round(n_head * target_ir ** (-r / (C - 1))))
        want = max(min_count, min(want, counts[ci]))
        idx_c = train_idx[y_tr == c]
        keep.append(rng.choice(idx_c, size=want, replace=False) if want < len(idx_c) else idx_c)
        target[int(c)] = want
    new_idx = np.sort(np.concatenate(keep))
    new_counts = np.array([target[int(c)] for c in classes])
    info = {"orig_counts": counts.tolist(), "new_counts": new_counts.tolist(),
            "orig_ir": float(counts.max() / counts.min()),
            "realised_ir": float(new_counts.max() / new_counts.min()),
            "target_ir": float(target_ir), "n_train_before": int(len(train_idx)),
            "n_train_after": int(len(new_idx))}
    return new_idx, info


def matched_scarcity_indices(train_idx, y_all, target_min_count, split_seed=0, min_count=3):
    """Control arm for X1: shrink EVERY class by the same fraction, so the
    imbalance ratio stays (almost) at its original value, while the rarest
    class ends with `target_min_count` training recordings.

    Use target_min_count = min(info['new_counts']) from extreme_train_indices
    for the same language and fold. Then the two arms have the same rarest
    keyword count but different IR, which separates data scarcity from IR.
    Same `split_seed` rule as above.
    """
    train_idx = np.asarray(train_idx)
    y_tr = np.asarray(y_all)[train_idx]
    classes, counts = np.unique(y_tr, return_counts=True)
    frac = min(1.0, target_min_count / counts.min())
    rng = np.random.default_rng(split_seed)
    keep, new_counts = [], []
    for c, n_c in zip(classes, counts):
        want = max(min_count, int(round(n_c * frac)))
        want = min(want, n_c)
        idx_c = train_idx[y_tr == c]
        keep.append(rng.choice(idx_c, size=want, replace=False) if want < n_c else idx_c)
        new_counts.append(want)
    new_idx = np.sort(np.concatenate(keep))
    new_counts = np.array(new_counts)
    info = {"orig_counts": counts.tolist(), "new_counts": new_counts.tolist(),
            "orig_ir": float(counts.max() / counts.min()),
            "realised_ir": float(new_counts.max() / new_counts.min()),
            "fraction_kept": float(frac), "n_train_before": int(len(train_idx)),
            "n_train_after": int(len(new_idx))}
    return new_idx, info


# ---------------------------------------------------------------------------
# X3: SMOTE in feature space (data-level, synthetic minority examples)
# ---------------------------------------------------------------------------
def smote_features(X_train, y_train, k=5, seed=0, target="max"):
    """Oversample every class up to the size of the largest class (target='max')
    with SMOTE (Chawla et al., 2002) on the cached features.

    New point = x + u * (x_nn - x), u ~ U(0, 1), x_nn one of the k nearest
    neighbours of x within the same class (Euclidean distance on the flattened
    feature). Works for (N, D) and (N, L, D). Apply to the TRAINING part of a
    fold only, never to validation or test. Then train with plain CE.

    Classes with a single example are duplicated (no neighbour exists).
    k is reduced to (class size - 1) when a class is smaller than k + 1.
    Returns X_aug, y_aug (originals first, synthetic examples appended).
    """
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    flat = X_train.reshape(len(X_train), -1).astype(np.float32)
    classes, counts = np.unique(y_train, return_counts=True)
    n_target = counts.max() if target == "max" else int(target)
    new_X, new_y = [X_train], [y_train]
    for c, n_c in zip(classes, counts):
        n_new = int(n_target - n_c)
        if n_new <= 0:
            continue
        idx = np.where(y_train == c)[0]
        Xc = flat[idx]
        if n_c == 1:
            synth = np.repeat(Xc, n_new, axis=0)
        else:
            kk = min(k, n_c - 1)
            # pairwise distances within the class (classes are small, so this is cheap)
            sq = (Xc ** 2).sum(1)
            d = sq[:, None] + sq[None, :] - 2.0 * Xc @ Xc.T
            np.fill_diagonal(d, np.inf)
            nn = np.argsort(d, axis=1)[:, :kk]                 # (n_c, kk)
            base = rng.integers(0, n_c, size=n_new)
            nb = nn[base, rng.integers(0, kk, size=n_new)]
            u = rng.random((n_new, 1), dtype=np.float32)
            synth = Xc[base] + u * (Xc[nb] - Xc[base])
        new_X.append(synth.reshape((n_new,) + X_train.shape[1:]).astype(X_train.dtype))
        new_y.append(np.full(n_new, c, dtype=y_train.dtype))
    return np.concatenate(new_X), np.concatenate(new_y)


# ---------------------------------------------------------------------------
# X3 (optional): Balanced-MixUp in feature space
# ---------------------------------------------------------------------------
def balanced_mixup(xb_inst, yb_inst, xb_bal, yb_bal, num_classes, alpha=0.2, rng=None):
    """Balanced-MixUp (Galdran et al., MICCAI 2021): mix a batch drawn with the
    normal (instance) sampler with a batch drawn with a class-balanced sampler.
    lam ~ Beta(alpha, 1) weights the BALANCED batch, so small alpha means a mild
    shift toward rare classes. Works on torch tensors.
    Train with soft-target CE:  loss = -(y_mix * log_softmax(logits)).sum(1).mean()
    """
    import torch
    rng = rng or np.random.default_rng()
    lam = float(rng.beta(alpha, 1.0))
    x = (1.0 - lam) * xb_inst + lam * xb_bal
    y1 = torch.nn.functional.one_hot(yb_inst, num_classes).float()
    y2 = torch.nn.functional.one_hot(yb_bal, num_classes).float()
    return x, (1.0 - lam) * y1 + lam * y2


# ---------------------------------------------------------------------------
# Self-test (no real data needed):  python imbalance_extras.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    counts = np.array([700, 520, 400, 330, 300, 260, 230, 210, 190, 170,
                       150, 140, 130, 120, 110, 100, 90, 80, 70, 60])  # IR ~ 11.7
    y_all = np.concatenate([np.full(n, c) for c, n in enumerate(counts)])
    X_all = rng.normal(size=(len(y_all), 25, 16)).astype(np.float32) + y_all[:, None, None] * 0.1
    train_idx = rng.permutation(len(y_all))[: int(0.7 * len(y_all))]

    for ir in (50, 100):
        a, info = extreme_train_indices(train_idx, y_all, ir, split_seed=1007)
        b, _ = extreme_train_indices(train_idx, y_all, ir, split_seed=1007)
        assert np.array_equal(a, b), "subset must be identical for the same split_seed"
        assert set(a) <= set(train_idx), "must only use training indices"
        print(f"X1 target IR {ir}: realised IR {info['realised_ir']:.1f}, "
              f"train {info['n_train_before']} -> {info['n_train_after']}, "
              f"tail counts {sorted(info['new_counts'])[:4]}")

        s, sinfo = matched_scarcity_indices(train_idx, y_all, min(info["new_counts"]), split_seed=1007)
        assert min(sinfo["new_counts"]) == min(info["new_counts"])
        print(f"   matched-scarcity arm: rarest {min(sinfo['new_counts'])}, IR {sinfo['realised_ir']:.1f} "
              f"(original {sinfo['orig_ir']:.1f}), train {sinfo['n_train_after']}")

    Xa, ya = smote_features(X_all[a], y_all[a], k=5, seed=11)
    cnt = np.bincount(ya)
    assert cnt.min() == cnt.max(), "SMOTE should balance every class"
    assert Xa.shape[1:] == X_all.shape[1:]
    # synthetic points lie on segments between same-class points: check range
    for c in range(20):
        orig = X_all[a][y_all[a] == c].reshape(-1, 400)
        syn = Xa[len(a):][ya[len(a):] == c].reshape(-1, 400)
        if len(syn):
            assert (syn.min(0) >= orig.min(0) - 1e-5).all() and (syn.max(0) <= orig.max(0) + 1e-5).all()
    print(f"X3 SMOTE: {len(a)} -> {len(ya)} training examples, every class = {cnt.max()}")

    try:
        import torch
        xi, yi = torch.randn(8, 25, 16), torch.randint(0, 20, (8,))
        xb, yb = torch.randn(8, 25, 16), torch.randint(0, 20, (8,))
        xm, ym = balanced_mixup(xi, yi, xb, yb, 20, alpha=0.2, rng=rng)
        assert torch.allclose(ym.sum(1), torch.ones(8))
        print("X3 Balanced-MixUp: OK")
    except ImportError:
        print("torch not installed: Balanced-MixUp test skipped")
    print("All self-tests passed.")
