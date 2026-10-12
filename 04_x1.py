# %%
from pathlib import Path
from collections import Counter
import copy
import random
import json
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

# Use cuda:1 here if you want the second GPU.
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Using:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(DEVICE))

# %%
ROOT = Path(
    './features'
)

# Change Job1 to Job2, Job3 or Job4 in the corresponding notebook.
OUTPUT_DIR = Path(
    './outputs/x1/Job1'
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "classification_results_all_methods.csv"
TUNING_PATH = OUTPUT_DIR / "selected_hyperparameters_all_methods.csv"

METHODS = ["CE", "WeightedCE", "ClassBalanced", "RandomOversampling_CE"]
# Trial first. For the full run: LANGUAGES = LOW_LANGS,
# RUN_FOLDS = [1, 2, 3, 4, 5], FINAL_SEEDS = [11, 22, 33].
FINAL_SEEDS = [11]
RUN_FOLDS = [1, 2, 3, 4, 5]
TUNING_SEED = 2026
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
DROPOUT = 0.30

# Small validation-only grids. Test data is never used to choose settings.
TUNING_GRIDS = {
    "Focal": [{"gamma": x} for x in (1.0, 2.0, 3.0)],
    "ClassBalanced": [{"beta": x} for x in (0.99, 0.999, 0.9999)],
    "LogitAdjustment": [{"tau": x} for x in (0.25, 0.5, 1.0)],
    "LDAM": [
        {"max_m": m, "drw": d}
        for m in (0.25, 0.5) for d in (False, True)
    ],
}

if not ROOT.is_dir():
    raise FileNotFoundError(f"Feature folder not found: {ROOT}")
#LOW_LANGS = ["rm-sursilv", "rm-vallader", "mt", "lv", "sah", ]
LOW_LANGS = ["rm-sursilv", "rm-vallader", "mt", "lv", "sah", "ka", "sl", "cv", "sk", "br", "ar", "ga-IE", "ia", "fy-NL", "ta", "cnh", "zh-CN", "ro", "lt", "el", "ha"]
MEDIUM_LANGS = ["uk", "tt", "mn", "ky", "et", "cs", "pt", "sv-SE", "tr", "nl", "id", "eo"]
HIGH_LANGS = ["it", "ru", "cy", "eu", "pl"]
ALL_LANGUAGES = LOW_LANGS + MEDIUM_LANGS + HIGH_LANGS
# Job 1
LANGUAGES = LOW_LANGS  # Set ALL_LANGUAGES for an optional 38-language extension.
# Verify these against the actual main-run batch size for each language
# before enabling the full run; the uploaded notebook specified these defaults.
# Original per-language values from the supervisor's final-day guide.
BATCH_SIZE_BY_LANGUAGE = {'ar': 256, 'br': 256, 'cnh': 256, 'cv': 256, 'el': 256, 'fy-NL': 256, 'ga-IE': 256, 'ha': 256, 'ia': 256, 'ka': 256, 'lt': 256, 'lv': 256, 'mt': 256, 'rm-sursilv': 256, 'rm-vallader': 256, 'ro': 256, 'sah': 256, 'sk': 256, 'sl': 256, 'ta': 256, 'zh-CN': 256, 'mn': 2048, 'nl': 2048, 'pt': 5096, 'sv-SE': 5096, 'tr': 5096, 'tt': 2048, 'uk': 2048, 'cs': 2048, 'eo': 5096, 'et': 2048, 'id': 2048, 'ky': 2048, 'cy': 5096, 'eu': 5096, 'it': 5096, 'pl': 5096, 'ru': 5096}
DRW_START_EPOCH = 50
assert len(ALL_LANGUAGES) == len(set(ALL_LANGUAGES)) == 38


# %%
import subprocess, sys
helper_path = Path("imbalance_extras.py")
helper_path.write_text('"""Drop-in helpers for the two quick follow-up experiments (X1 and X3).\n\nBoth work on the CACHED XLS-R features, so no audio needs to be re-encoded.\nCall them inside the existing loop (language -> fold -> seed). Do not change\nanything else (encoder, pooling, MLP, optimizer, early stopping, metrics, or\nthe outputs.npz / per_class_metrics.csv / bottom20_metrics.csv files).\n\nX1  Extreme imbalance  : extreme_train_indices(...)   -> subsample the TRAIN part only\nX3  Synthetic minority : smote_features(...)          -> SMOTE in feature space (TRAIN part only)\n    (optional)         : balanced_mixup(...)          -> Balanced-MixUp in feature space\n\nShapes: X may be (N, D) or (N, L, D) (per-layer pooled features). Both work.\n"""\nimport numpy as np\n\n# ---------------------------------------------------------------------------\n# X1: extreme imbalance by subsampling the training split\n# ---------------------------------------------------------------------------\ndef extreme_train_indices(train_idx, y_all, target_ir, split_seed=0, min_count=3):\n    """Return a subset of `train_idx` whose class counts follow an exponential\n    long tail with max/min ratio close to `target_ir`.\n\n    - Classes are ranked by their ORIGINAL training count (largest = rank 0).\n    - Target count for rank r: n_head * target_ir ** (-r / (C-1)), never more\n      than the class actually has, and never fewer than `min_count`.\n    - The head class keeps all its training recordings.\n    - Validation and test indices are NOT touched.\n    - `split_seed` must be the same for every method and training seed of a\n      given language and fold, so that all methods see the SAME subset.\n      Use split_seed = 1000 * fold + 7 (independent of the training seed).\n\n    Returns (new_train_idx, info) where info has the realised counts and IR.\n    """\n    train_idx = np.asarray(train_idx)\n    y_tr = np.asarray(y_all)[train_idx]\n    classes, counts = np.unique(y_tr, return_counts=True)\n    C = len(classes)\n    order = np.argsort(-counts, kind="stable")          # rank 0 = most frequent\n    n_head = counts[order[0]]\n    rng = np.random.default_rng(split_seed)\n    keep = []\n    target = {}\n    for r, ci in enumerate(order):\n        c = classes[ci]\n        want = int(round(n_head * target_ir ** (-r / (C - 1))))\n        want = max(min_count, min(want, counts[ci]))\n        idx_c = train_idx[y_tr == c]\n        keep.append(rng.choice(idx_c, size=want, replace=False) if want < len(idx_c) else idx_c)\n        target[int(c)] = want\n    new_idx = np.sort(np.concatenate(keep))\n    new_counts = np.array([target[int(c)] for c in classes])\n    info = {"orig_counts": counts.tolist(), "new_counts": new_counts.tolist(),\n            "orig_ir": float(counts.max() / counts.min()),\n            "realised_ir": float(new_counts.max() / new_counts.min()),\n            "target_ir": float(target_ir), "n_train_before": int(len(train_idx)),\n            "n_train_after": int(len(new_idx))}\n    return new_idx, info\n\n\ndef matched_scarcity_indices(train_idx, y_all, target_min_count, split_seed=0, min_count=3):\n    """Control arm for X1: shrink EVERY class by the same fraction, so the\n    imbalance ratio stays (almost) at its original value, while the rarest\n    class ends with `target_min_count` training recordings.\n\n    Use target_min_count = min(info[\'new_counts\']) from extreme_train_indices\n    for the same language and fold. Then the two arms have the same rarest\n    keyword count but different IR, which separates data scarcity from IR.\n    Same `split_seed` rule as above.\n    """\n    train_idx = np.asarray(train_idx)\n    y_tr = np.asarray(y_all)[train_idx]\n    classes, counts = np.unique(y_tr, return_counts=True)\n    frac = min(1.0, target_min_count / counts.min())\n    rng = np.random.default_rng(split_seed)\n    keep, new_counts = [], []\n    for c, n_c in zip(classes, counts):\n        want = max(min_count, int(round(n_c * frac)))\n        want = min(want, n_c)\n        idx_c = train_idx[y_tr == c]\n        keep.append(rng.choice(idx_c, size=want, replace=False) if want < n_c else idx_c)\n        new_counts.append(want)\n    new_idx = np.sort(np.concatenate(keep))\n    new_counts = np.array(new_counts)\n    info = {"orig_counts": counts.tolist(), "new_counts": new_counts.tolist(),\n            "orig_ir": float(counts.max() / counts.min()),\n            "realised_ir": float(new_counts.max() / new_counts.min()),\n            "fraction_kept": float(frac), "n_train_before": int(len(train_idx)),\n            "n_train_after": int(len(new_idx))}\n    return new_idx, info\n\n\n# ---------------------------------------------------------------------------\n# X3: SMOTE in feature space (data-level, synthetic minority examples)\n# ---------------------------------------------------------------------------\ndef smote_features(X_train, y_train, k=5, seed=0, target="max"):\n    """Oversample every class up to the size of the largest class (target=\'max\')\n    with SMOTE (Chawla et al., 2002) on the cached features.\n\n    New point = x + u * (x_nn - x), u ~ U(0, 1), x_nn one of the k nearest\n    neighbours of x within the same class (Euclidean distance on the flattened\n    feature). Works for (N, D) and (N, L, D). Apply to the TRAINING part of a\n    fold only, never to validation or test. Then train with plain CE.\n\n    Classes with a single example are duplicated (no neighbour exists).\n    k is reduced to (class size - 1) when a class is smaller than k + 1.\n    Returns X_aug, y_aug (originals first, synthetic examples appended).\n    """\n    rng = np.random.default_rng(seed)\n    X_train = np.asarray(X_train)\n    y_train = np.asarray(y_train)\n    flat = X_train.reshape(len(X_train), -1).astype(np.float32)\n    classes, counts = np.unique(y_train, return_counts=True)\n    n_target = counts.max() if target == "max" else int(target)\n    new_X, new_y = [X_train], [y_train]\n    for c, n_c in zip(classes, counts):\n        n_new = int(n_target - n_c)\n        if n_new <= 0:\n            continue\n        idx = np.where(y_train == c)[0]\n        Xc = flat[idx]\n        if n_c == 1:\n            synth = np.repeat(Xc, n_new, axis=0)\n        else:\n            kk = min(k, n_c - 1)\n            # pairwise distances within the class (classes are small, so this is cheap)\n            sq = (Xc ** 2).sum(1)\n            d = sq[:, None] + sq[None, :] - 2.0 * Xc @ Xc.T\n            np.fill_diagonal(d, np.inf)\n            nn = np.argsort(d, axis=1)[:, :kk]                 # (n_c, kk)\n            base = rng.integers(0, n_c, size=n_new)\n            nb = nn[base, rng.integers(0, kk, size=n_new)]\n            u = rng.random((n_new, 1), dtype=np.float32)\n            synth = Xc[base] + u * (Xc[nb] - Xc[base])\n        new_X.append(synth.reshape((n_new,) + X_train.shape[1:]).astype(X_train.dtype))\n        new_y.append(np.full(n_new, c, dtype=y_train.dtype))\n    return np.concatenate(new_X), np.concatenate(new_y)\n\n\n# ---------------------------------------------------------------------------\n# X3 (optional): Balanced-MixUp in feature space\n# ---------------------------------------------------------------------------\ndef balanced_mixup(xb_inst, yb_inst, xb_bal, yb_bal, num_classes, alpha=0.2, rng=None):\n    """Balanced-MixUp (Galdran et al., MICCAI 2021): mix a batch drawn with the\n    normal (instance) sampler with a batch drawn with a class-balanced sampler.\n    lam ~ Beta(alpha, 1) weights the BALANCED batch, so small alpha means a mild\n    shift toward rare classes. Works on torch tensors.\n    Train with soft-target CE:  loss = -(y_mix * log_softmax(logits)).sum(1).mean()\n    """\n    import torch\n    rng = rng or np.random.default_rng()\n    lam = float(rng.beta(alpha, 1.0))\n    x = (1.0 - lam) * xb_inst + lam * xb_bal\n    y1 = torch.nn.functional.one_hot(yb_inst, num_classes).float()\n    y2 = torch.nn.functional.one_hot(yb_bal, num_classes).float()\n    return x, (1.0 - lam) * y1 + lam * y2\n\n\n# ---------------------------------------------------------------------------\n# Self-test (no real data needed):  python imbalance_extras.py\n# ---------------------------------------------------------------------------\nif __name__ == "__main__":\n    rng = np.random.default_rng(0)\n    counts = np.array([700, 520, 400, 330, 300, 260, 230, 210, 190, 170,\n                       150, 140, 130, 120, 110, 100, 90, 80, 70, 60])  # IR ~ 11.7\n    y_all = np.concatenate([np.full(n, c) for c, n in enumerate(counts)])\n    X_all = rng.normal(size=(len(y_all), 25, 16)).astype(np.float32) + y_all[:, None, None] * 0.1\n    train_idx = rng.permutation(len(y_all))[: int(0.7 * len(y_all))]\n\n    for ir in (50, 100):\n        a, info = extreme_train_indices(train_idx, y_all, ir, split_seed=1007)\n        b, _ = extreme_train_indices(train_idx, y_all, ir, split_seed=1007)\n        assert np.array_equal(a, b), "subset must be identical for the same split_seed"\n        assert set(a) <= set(train_idx), "must only use training indices"\n        print(f"X1 target IR {ir}: realised IR {info[\'realised_ir\']:.1f}, "\n              f"train {info[\'n_train_before\']} -> {info[\'n_train_after\']}, "\n              f"tail counts {sorted(info[\'new_counts\'])[:4]}")\n\n        s, sinfo = matched_scarcity_indices(train_idx, y_all, min(info["new_counts"]), split_seed=1007)\n        assert min(sinfo["new_counts"]) == min(info["new_counts"])\n        print(f"   matched-scarcity arm: rarest {min(sinfo[\'new_counts\'])}, IR {sinfo[\'realised_ir\']:.1f} "\n              f"(original {sinfo[\'orig_ir\']:.1f}), train {sinfo[\'n_train_after\']}")\n\n    Xa, ya = smote_features(X_all[a], y_all[a], k=5, seed=11)\n    cnt = np.bincount(ya)\n    assert cnt.min() == cnt.max(), "SMOTE should balance every class"\n    assert Xa.shape[1:] == X_all.shape[1:]\n    # synthetic points lie on segments between same-class points: check range\n    for c in range(20):\n        orig = X_all[a][y_all[a] == c].reshape(-1, 400)\n        syn = Xa[len(a):][ya[len(a):] == c].reshape(-1, 400)\n        if len(syn):\n            assert (syn.min(0) >= orig.min(0) - 1e-5).all() and (syn.max(0) <= orig.max(0) + 1e-5).all()\n    print(f"X3 SMOTE: {len(a)} -> {len(ya)} training examples, every class = {cnt.max()}")\n\n    try:\n        import torch\n        xi, yi = torch.randn(8, 25, 16), torch.randint(0, 20, (8,))\n        xb, yb = torch.randn(8, 25, 16), torch.randint(0, 20, (8,))\n        xm, ym = balanced_mixup(xi, yi, xb, yb, 20, alpha=0.2, rng=rng)\n        assert torch.allclose(ym.sum(1), torch.ones(8))\n        print("X3 Balanced-MixUp: OK")\n    except ImportError:\n        print("torch not installed: Balanced-MixUp test skipped")\n    print("All self-tests passed.")\n', encoding="utf-8")
subprocess.run([sys.executable, str(helper_path)], check=True)


# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureDataset(Dataset):
    def __init__(self, data):
        self.X = data["layer_features"]
        self.y = data["labels"].long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index].float(), self.y[index]


def load_split(lang_dir, split):
    path = lang_dir / f"{split}.pt"

    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")

    data = torch.load(path, map_location="cpu")

    required = {"layer_features", "labels", "class_names"}

    if not required.issubset(data):
        raise ValueError(
            f"{path} must contain layer_features, labels "
            "and class_names."
        )

    return data

# %%
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5)
        )

    def forward(self, x):
        # Normalise features and each class weight vector.
        features = nn.functional.normalize(x, p=2, dim=1)
        weights = nn.functional.normalize(
            self.weight, p=2, dim=1
        )

        # Return unscaled cosine scores.
        return nn.functional.linear(features, weights)


class WAPMLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        num_classes,
        use_cosine=False,
        logit_scale=1.0,
    ):
        super().__init__()

        self.layer_logits = nn.Parameter(
            torch.zeros(num_layers)
        )

        self.logit_scale = float(logit_scale)

        final_layer = (
            NormedLinear(512, num_classes)
            if use_cosine
            else nn.Linear(512, num_classes)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            final_layer,
        )

    def forward(self, x):
        layer_weights = torch.softmax(
            self.layer_logits, dim=0
        )

        pooled = (
            x * layer_weights[None, :, None]
        ).sum(dim=1)

        return self.classifier(pooled)

# %%
def inverse_weights(counts):
    weights = 1.0 / torch.as_tensor(counts, dtype=torch.float32).clamp_min(1)
    return weights / weights.mean()


def effective_number_weights(counts, beta):
    counts = np.asarray(counts, dtype=np.float64)
    weights = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), 1e-12)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_loss(method, logits, y, counts, cfg, epoch):
    counts_t = torch.as_tensor(counts, dtype=torch.float32, device=logits.device)
    if method in ("CE", "RandomOversampling_CE"):
        return nn.functional.cross_entropy(logits, y)
    if method == "WeightedCE":
        return nn.functional.cross_entropy(
            logits, y, weight=inverse_weights(counts).to(logits.device)
        )
    if method == "Focal":
        ce = nn.functional.cross_entropy(logits, y, reduction="none")
        return (((1 - torch.exp(-ce)) ** cfg["gamma"]) * ce).mean()
    if method == "ClassBalanced":
        weights = effective_number_weights(counts, cfg["beta"]).to(logits.device)
        return nn.functional.cross_entropy(logits, y, weight=weights)
    if method == "LogitAdjustment":
        priors = counts_t / counts_t.sum()
        adjusted = logits + cfg["tau"] * torch.log(priors)[None, :]
        return nn.functional.cross_entropy(adjusted, y)
    if method == "BalancedSoftmax":
        return nn.functional.cross_entropy(logits + torch.log(counts_t)[None, :], y)
    if method == "LDAM":
        margins = 1.0 / np.sqrt(np.sqrt(np.asarray(counts, dtype=np.float64)))
        margins = margins * (cfg["max_m"] / margins.max())
        margins = torch.tensor(margins, dtype=logits.dtype, device=logits.device)
        adjusted = logits.clone()
        adjusted[torch.arange(len(y), device=logits.device), y] -= margins[y]
        adjusted = cfg.get("scale", 30.0) * adjusted
        if cfg.get("drw", False) and epoch >= DRW_START_EPOCH:
            weights = effective_number_weights(
                counts, cfg.get("beta", 0.9999)
            ).to(logits.device)
            return nn.functional.cross_entropy(adjusted, y, weight=weights)
        return nn.functional.cross_entropy(adjusted, y)
    raise ValueError(f"Unknown method: {method}")


# %%
def make_loaders(train_data, val_data, test_data, counts, seed, oversample=False):
    train_ds = FeatureDataset(train_data)
    val_ds = FeatureDataset(val_data)
    test_ds = FeatureDataset(test_data)
    generator = torch.Generator().manual_seed(seed)

    if oversample:
        sample_weights = torch.tensor(
            [1.0 / counts[int(y)] for y in train_data["labels"]],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights),
            replacement=True, generator=generator
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator
        )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


# %%
@torch.inference_mode()
def evaluate(model, loader, counts):
    model.eval()

    prob_parts = []
    label_parts = []

    for x, y in loader:
        # Inference uses scale 30 without a target-class margin.
        logits = model(x.to(DEVICE)) * model.logit_scale

        prob_parts.append(
            torch.softmax(logits, dim=1).cpu().numpy()
        )
        label_parts.append(y.numpy())

    probs = np.concatenate(prob_parts)
    y_true = np.concatenate(label_parts)
    y_pred = probs.argmax(axis=1)

    class_ids = np.arange(len(counts))

    _, recalls, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )

    rare_n = max(1, int(math.ceil(0.20 * len(counts))))
    rare_ids = np.argsort(np.asarray(counts))[:rare_n]

    y_bin = label_binarize(y_true, classes=class_ids)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0,
        ),
        "micro_f1": f1_score(
            y_true, y_pred,
            labels=class_ids,
            average="micro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_auprc": average_precision_score(
            y_bin, probs, average="macro"
        ),
        "micro_auprc": average_precision_score(
            y_bin, probs, average="micro"
        ),
        "bottom20_recall": float(recalls[rare_ids].mean()),
    }


def train_model(
    train_loader,
    val_loader,
    method,
    counts,
    cfg,
    seed,
):
    set_seed(seed)

    example_x, _ = next(iter(train_loader))

    model = WAPMLP(
        num_layers=example_x.shape[1],
        input_dim=example_x.shape[2],
        num_classes=len(counts),
        use_cosine=(method == "LDAM"),
        logit_scale=cfg.get("scale", 30.0) if method == "LDAM" else 1.0,
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_score = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    drw_run = method == "LDAM" and cfg.get("drw", False)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # Unscaled cosine scores go into the loss.
            loss = compute_loss(
                method,
                model(x),
                y,
                counts,
                cfg,
                epoch,
            )

            loss.backward()
            optimizer.step()

        val_score = evaluate(
            model, val_loader, counts
        )["macro_f1"]

        print(
            f"{method} {cfg} epoch {epoch}: "
            f"validation Macro-F1={val_score:.4f}"
        )

        # Save the best checkpoint across all epochs.
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            # DRW patience starts only from epoch 50.
            if not drw_run or epoch >= DRW_START_EPOCH:
                wait += 1
            else:
                wait = 0

        if wait >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")

    model.load_state_dict(best_state)

    return model, best_score, best_epoch
    

# %%
def bottom20_details(y_true, y_pred, counts):
    class_ids = np.arange(len(counts))

    rare_n = max(1, int(math.ceil(0.20 * len(counts))))
    rare_ids = np.argsort(np.asarray(counts))[:rare_n]

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_ids,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids,
    )

    return rare_ids, precision, recall, f1, support, cm


@torch.inference_mode()
def save_run_outputs(
    model,
    va,
    te,
    counts,
    class_names,
    run_dir,
    val_idx,
    test_idx,
    fit_idx,
):
    model.eval()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def collect(loader):
        logits_parts = []
        label_parts = []

        for x, y in loader:
            logits = model(x.to(DEVICE)) * model.logit_scale

            logits_parts.append(logits.cpu().numpy())
            label_parts.append(y.numpy())

        return (
            np.concatenate(logits_parts),
            np.concatenate(label_parts),
        )

    val_logits, val_labels = collect(va)
    test_logits, test_labels = collect(te)

    pred = (
        torch.softmax(torch.from_numpy(test_logits), dim=1)
        .numpy()
        .argmax(axis=1)
    )

    rare_ids, precision, recall, f1, support, cm = (
        bottom20_details(test_labels, pred, counts)
    )

    names = np.asarray(class_names, dtype=str)
    counts_array = np.asarray(counts)

    np.savez(
        run_dir / "outputs.npz",
        val_logits=val_logits,
        val_labels=val_labels,
        test_logits=test_logits,
        test_labels=test_labels,
        test_predictions=pred,
        train_class_counts=counts_array,
        class_names=names,
        bottom20_class_ids=rare_ids,
        confusion_matrix=cm,
        bottom20_confusion_rows=cm[rare_ids, :],
        bottom20_confusion_columns=cm[:, rare_ids],
        train_indices=fit_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )

    pd.DataFrame({
        "class_id": np.arange(len(counts)),
        "keyword": names,
        "train_count": counts_array,
        "test_support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "is_bottom20": np.isin(
            np.arange(len(counts)), rare_ids
        ),
    }).to_csv(
        run_dir / "per_class_metrics.csv",
        index=False,
    )

    pd.DataFrame({
        "class_id": rare_ids,
        "keyword": names[rare_ids],
        "train_count": counts_array[rare_ids],
        "test_support": support[rare_ids],
        "precision": precision[rare_ids],
        "recall": recall[rare_ids],
        "f1": f1[rare_ids],
    }).to_csv(
        run_dir / "bottom20_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        cm,
        index=names,
        columns=names,
    ).rename_axis("true_keyword").to_csv(
        run_dir / "confusion_matrix_full.csv"
    )

    pd.DataFrame(
        cm[rare_ids, :],
        index=names[rare_ids],
        columns=names,
    ).rename_axis("true_keyword").to_csv(
        run_dir / "bottom20_confusion_rows.csv"
    )

    pd.DataFrame(
        cm[:, rare_ids],
        index=names,
        columns=names[rare_ids],
    ).rename_axis("true_keyword").to_csv(
        run_dir / "bottom20_confusion_columns.csv"
    )

    return {
        "bottom20_precision": float(
            precision[rare_ids].mean()
        ),
        "bottom20_recall": float(
            recall[rare_ids].mean()
        ),
        "bottom20_f1": float(
            f1[rare_ids].mean()
        ),
    }

# %%
from IPython.display import display
# X1 pilot: same real training subset for every method and seed in each arm.
import time
from imbalance_extras import extreme_train_indices, matched_scarcity_indices

N_FOLDS = 5
CV_SEED = 42
VALIDATION_FRACTION_OF_OUTER_TRAIN = 0.125
METHOD_NAMES = {"CE": "CE", "WeightedCE": "Weighted CE",
                "ClassBalanced": "Class-Balanced", "RandomOversampling_CE": "ROS + CE"}

assert LANGUAGES
assert len(LANGUAGES) == len(set(LANGUAGES))
assert set(LANGUAGES).issubset(ALL_LANGUAGES)




assert RUN_FOLDS == [1, 2, 3, 4, 5] and FINAL_SEEDS == [11]
expected_final = len(LANGUAGES) * len(RUN_FOLDS) * len(FINAL_SEEDS) * len(METHODS) * 2
expected_tuning = len(LANGUAGES) * len(RUN_FOLDS) * len(TUNING_GRIDS["ClassBalanced"]) * 2
for lang in LANGUAGES:
    for split in ("train", "val", "test"):
        if not (ROOT / lang / f"{split}.pt").is_file():
            raise FileNotFoundError(ROOT / lang / f"{split}.pt")
for arm in ("X1_IR100", "X1_SCARCE"):
    arm_dir = OUTPUT_DIR / arm
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise FileExistsError(f"Existing outputs: {arm_dir}; use a fresh OUTPUT_DIR.")
print("Final trainings:", expected_final, "CB tuning trainings:", expected_tuning)
if RESULTS_PATH.exists() or TUNING_PATH.exists():
    raise FileExistsError("Existing pilot results: select a fresh OUTPUT_DIR; keep completed files.")

def select_rows(data, indices):
    ix = torch.as_tensor(indices, dtype=torch.long)
    return {"layer_features": data["layer_features"].index_select(0, ix),
            "labels": data["labels"].index_select(0, ix), "class_names": data["class_names"]}

def sync_gpu():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)

result_rows, tuning_rows, pilot_timings = [], [], []
pilot_start = time.perf_counter()
for lang in LANGUAGES:
    BATCH_SIZE = BATCH_SIZE_BY_LANGUAGE[lang]
    parts = [load_split(ROOT / lang, split) for split in ("train", "val", "test")]
    class_names = parts[0]["class_names"]
    if any(list(part["class_names"]) != list(class_names) for part in parts[1:]):
        raise ValueError("Class mappings differ across saved splits.")
    features = torch.cat([part["layer_features"] for part in parts], dim=0)
    labels = torch.cat([part["labels"] for part in parts], dim=0).long()
    y = labels.numpy()
    C = len(class_names)
    if features.ndim != 3 or np.bincount(y, minlength=C).min() < 5:
        raise ValueError("Expected (N, layers, D) features and >=5 recordings per keyword.")
    data = {"layer_features": features, "labels": labels, "class_names": class_names}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (outer_train, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        if fold not in RUN_FOLDS:
            continue
        fit_idx, val_idx = train_test_split(outer_train, test_size=0.125,
                                            random_state=42+fold, stratify=y[outer_train])
        split_seed = 1000 * fold + 7
        train_A, info_A = extreme_train_indices(fit_idx, y, target_ir=100, split_seed=split_seed)
        train_B, info_B = matched_scarcity_indices(
            fit_idx, y, min(info_A["new_counts"]), split_seed=split_seed)
        assert min(info_A["new_counts"]) == min(info_B["new_counts"])
        repeat_A, _ = extreme_train_indices(fit_idx, y, target_ir=100, split_seed=split_seed)
        assert np.array_equal(train_A, repeat_A)
        val_data, test_data = select_rows(data, val_idx), select_rows(data, test_idx)
        for arm, train_idx, info in [("X1_IR100", train_A, info_A),
                                      ("X1_SCARCE", train_B, info_B)]:
            sync_gpu()
            arm_start = time.perf_counter()
            assert np.isin(train_idx, fit_idx).all()
            assert not np.intersect1d(train_idx, val_idx).size
            assert not np.intersect1d(train_idx, test_idx).size
            n_full = len(fit_idx)
            n_sub = len(train_idx)
            base_batch_size = BATCH_SIZE_BY_LANGUAGE[lang]
            BATCH_SIZE = max(8, round(base_batch_size * n_sub / n_full))
            print(f"{lang} fold={fold} {arm}: full={n_full}, subset={n_sub}, "
                  f"batch={BATCH_SIZE}, updates/epoch={math.ceil(n_sub / BATCH_SIZE)}")
            train_data = select_rows(data, train_idx)
            counts = np.bincount(y[train_idx], minlength=C).tolist()
            assert counts == info["new_counts"]
            fold_dir = OUTPUT_DIR / arm / lang / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            metadata = {**info, "language": lang, "fold": fold,
                        "split_seed": split_seed, "class_names": list(class_names),
                        "batch_size": BATCH_SIZE, "base_batch_size": base_batch_size,
                        "updates_per_epoch": math.ceil(n_sub / BATCH_SIZE)}
            (fold_dir / "subsample_info.json").write_text(json.dumps(metadata, indent=2))
            np.savez(fold_dir / "subset_indices.npz", original_train_indices=fit_idx,
                     train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)
            print(f"{arm}: original IR={info['orig_ir']:.3f}, "
                  f"realised IR={info['realised_ir']:.3f}, counts={counts}")
            for method in METHODS:
                sync_gpu()
                method_start = time.perf_counter()
                best_cfg, tuning_score, tuning_epoch = {}, None, None
                # Only Class-Balanced has a grid among the core methods.
                if method == "ClassBalanced":
                    best_score = -1.0
                    for cfg in TUNING_GRIDS["ClassBalanced"]:
                        tr, va, te = make_loaders(train_data, val_data, test_data, counts,
                                                  TUNING_SEED+fold, oversample=False)
                        candidate, score, epoch = train_model(
                            tr, va, method, counts, cfg, TUNING_SEED+fold)
                        if score > best_score:
                            best_score, best_cfg = score, dict(cfg)
                            tuning_score, tuning_epoch = score, epoch
                        del candidate, tr, va, te
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()
                for seed in FINAL_SEEDS:
                    sync_gpu()
                    final_start = time.perf_counter()
                    tr, va, te = make_loaders(train_data, val_data, test_data, counts, seed,
                                              oversample=(method == "RandomOversampling_CE"))
                    model, val_f1, best_epoch = train_model(tr, va, method, counts, best_cfg, seed)
                    run_dir = OUTPUT_DIR / arm / lang / METHOD_NAMES[method] / f"fold_{fold}" / f"seed_{seed}"
                    detailed = save_run_outputs(model, va, te, counts, class_names,
                                                run_dir, val_idx, test_idx, train_idx)
                    metrics = evaluate(model, te, counts)
                    metrics.update(detailed)
                    torch.save({"model_state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                                "method": METHOD_NAMES[method], "arm": arm,
                                "selected_config": best_cfg, "fold": fold, "seed": seed,
                                "best_epoch": best_epoch, "best_validation_macro_f1": val_f1,
                                "class_names": class_names, "train_class_counts": counts,
                                "train_indices": torch.as_tensor(train_idx),
                                "val_indices": torch.as_tensor(val_idx), "test_indices": torch.as_tensor(test_idx),
                                "num_layers": features.shape[1], "input_dim": features.shape[2],
                                "dropout": DROPOUT, "logit_scale": model.logit_scale,
                                "batch_size": BATCH_SIZE,
                                "use_cosine": False}, run_dir / "best_model.pt")
                    with np.load(run_dir / "outputs.npz", allow_pickle=False) as out:
                        assert np.array_equal(out["train_class_counts"], counts)
                        assert np.array_equal(out["train_indices"], train_idx)
                        assert np.array_equal(out["val_labels"], y[val_idx])
                        assert np.array_equal(out["test_labels"], y[test_idx])
                    sync_gpu()
                    final_seconds = time.perf_counter() - final_start
                    timing = {"final_run_seconds": final_seconds,
                              "method_seconds_including_tuning": time.perf_counter()-method_start,
                              "batch_size": BATCH_SIZE, "updates_per_epoch": math.ceil(n_sub / BATCH_SIZE)}
                    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
                    result_rows.append({"arm": arm, "language": lang, "method": METHOD_NAMES[method],
                                        "fold": fold, "seed": seed, "selected_config": json.dumps(best_cfg),
                                        "batch_size": BATCH_SIZE, "realised_ir": info["realised_ir"],
                                        "n_train": len(train_idx), "best_epoch": best_epoch,
                                        "best_validation_macro_f1": val_f1, **timing, **metrics})
                    tuning_rows.append({"arm": arm, "language": lang, "method": METHOD_NAMES[method],
                                        "fold": fold, "seed": seed, "selected_config": json.dumps(best_cfg),
                                        "tuning_seed": TUNING_SEED+fold if method=="ClassBalanced" else None,
                                        "tuning_validation_macro_f1": tuning_score, "tuning_best_epoch": tuning_epoch})
                    pd.DataFrame(result_rows).to_csv(RESULTS_PATH, index=False)
                    pd.DataFrame(tuning_rows).to_csv(TUNING_PATH, index=False)
                    pd.DataFrame(result_rows).to_excel(OUTPUT_DIR / "classification_results.xlsx", index=False)
                    pd.DataFrame(tuning_rows).to_excel(OUTPUT_DIR / "selected_hyperparameters.xlsx", index=False)
                    print(f"{arm} {METHOD_NAMES[method]}: Macro-F1={metrics['macro_f1']:.4f}; "
                          f"bottom20 recall={metrics['bottom20_recall']:.4f}; final={final_seconds/60:.2f} min")
                    del model, tr, va, te
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
            sync_gpu()
            pilot_timings.append({"arm": arm, "seconds_including_tuning": time.perf_counter()-arm_start})
summary = {"total_seconds": time.perf_counter()-pilot_start,
           "final_trainings": len(result_rows), "expected_final_trainings": expected_final,
           "additional_cb_tuning_trainings": expected_tuning, "arm_timings": pilot_timings}
(OUTPUT_DIR / "experiment_timing.json").write_text(json.dumps(summary, indent=2))
print(f"Completed {len(result_rows)}/{expected_final} final runs; total={summary['total_seconds']/60:.2f} min")
display(pd.DataFrame(result_rows)[["arm", "language", "fold", "seed", "method", "batch_size", "best_epoch", "macro_f1", "macro_auprc", "bottom20_recall"]])


# %%
