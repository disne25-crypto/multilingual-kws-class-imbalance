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

# ============================================================
# SMOTE + CE SETTINGS
# Five languages × five folds × three seeds = 75 runs
# ============================================================

from pathlib import Path

ROOT = Path(
    './features'
)

OUTPUT_DIR = Path(
    './outputs/smote'
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "classification_results_all_methods.csv"
TUNING_PATH = OUTPUT_DIR / "selected_hyperparameters_all_methods.csv"

METHODS = ["SMOTE + CE"]
FINAL_SEEDS = [11, 22, 33]
RUN_FOLDS = [1, 2, 3, 4, 5]

TUNING_SEED = 2026
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
DROPOUT = 0.30

# No hyperparameter search for SMOTE + CE.
TUNING_GRIDS = {}

LOW_LANGS = [
    "rm-sursilv", "rm-vallader", "mt", "lv", "sah",
    "ka", "sl", "cv", "sk", "br", "ar", "ga-IE",
    "ia", "fy-NL", "ta", "cnh", "zh-CN", "ro",
    "lt", "el", "ha",
]

MEDIUM_LANGS = [
    "uk", "tt", "mn", "ky", "et", "cs",
    "pt", "sv-SE", "tr", "nl", "id", "eo",
]

HIGH_LANGS = ["it", "ru", "cy", "eu", "pl"]

ALL_LANGUAGES = LOW_LANGS + MEDIUM_LANGS + HIGH_LANGS

# Selected languages for this run.
LANGUAGES = LOW_LANGS  # Set ALL_LANGUAGES for an optional 38-language extension.

# Selected low-resource languages use batch size 256.
# Verify other languages against your main runs before enabling them.
BATCH_SIZE_BY_LANGUAGE = {'ar': 256, 'br': 256, 'cnh': 256, 'cv': 256, 'el': 256, 'fy-NL': 256, 'ga-IE': 256, 'ha': 256, 'ia': 256, 'ka': 256, 'lt': 256, 'lv': 256, 'mt': 256, 'rm-sursilv': 256, 'rm-vallader': 256, 'ro': 256, 'sah': 256, 'sk': 256, 'sl': 256, 'ta': 256, 'zh-CN': 256, 'mn': 2048, 'nl': 2048, 'pt': 5096, 'sv-SE': 5096, 'tr': 5096, 'tt': 2048, 'uk': 2048, 'cs': 2048, 'eo': 5096, 'et': 2048, 'id': 2048, 'ky': 2048, 'cy': 5096, 'eu': 5096, 'it': 5096, 'pl': 5096, 'ru': 5096}

DRW_START_EPOCH = 50

assert len(ALL_LANGUAGES) == len(set(ALL_LANGUAGES)) == 38
assert len(LANGUAGES) == len(set(LANGUAGES))
assert set(LANGUAGES).issubset(ALL_LANGUAGES)

if not ROOT.is_dir():
    raise FileNotFoundError(f"Feature folder not found: {ROOT}")

for lang in LANGUAGES:
    for split in ("train", "val", "test"):
        feature_path = ROOT / lang / f"{split}.pt"
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Missing feature file: {feature_path}"
            )

expected_runs = (
    len(LANGUAGES) * len(RUN_FOLDS) * len(FINAL_SEEDS)
)

print("Method:", METHODS)
print("Languages:", LANGUAGES)
print("Folds:", RUN_FOLDS)
print("Seeds:", FINAL_SEEDS)
print("Expected trainings:", expected_runs)
print("Save all results inside:", OUTPUT_DIR)

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
    counts_after_smote,
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
        train_class_counts_after_smote=np.asarray(counts_after_smote),
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
# SMOTE + CE only; no extra tuning training.
import time
from imbalance_extras import smote_features

N_FOLDS = 5
CV_SEED = 42
VALIDATION_FRACTION_OF_OUTER_TRAIN = 0.125
assert RUN_FOLDS and set(RUN_FOLDS).issubset({1, 2, 3, 4, 5})
assert FINAL_SEEDS and set(FINAL_SEEDS).issubset({11, 22, 33})
if RESULTS_PATH.exists() or TUNING_PATH.exists():
    raise FileExistsError("Results already exist. Use a new OUTPUT_DIR for a fresh run.")

def select_rows(data, indices):
    ix = torch.as_tensor(indices, dtype=torch.long)
    return {"layer_features": data["layer_features"].index_select(0, ix),
            "labels": data["labels"].index_select(0, ix),
            "class_names": data["class_names"]}

result_rows, tuning_rows = [], []
experiment_start = time.perf_counter()
for lang in LANGUAGES:
    BATCH_SIZE = BATCH_SIZE_BY_LANGUAGE[lang]
    saved_splits = [load_split(ROOT / lang, split) for split in ("train", "val", "test")]
    class_names = saved_splits[0]["class_names"]
    if any(list(p["class_names"]) != list(class_names) for p in saved_splits[1:]):
        raise ValueError(f"Class mappings differ: {lang}")
    all_features = torch.cat([p["layer_features"] for p in saved_splits], dim=0)
    all_labels = torch.cat([p["labels"] for p in saved_splits], dim=0).long()
    labels_np = all_labels.numpy()
    n_classes = len(class_names)
    if labels_np.min() != 0 or labels_np.max() != n_classes - 1:
        raise ValueError("Labels must be encoded 0 through n_classes-1.")
    if all_features.ndim != 3:
        raise ValueError("This WAPMLP requires features shaped (N, layers, D).")
    full_counts = np.bincount(labels_np, minlength=n_classes)
    if full_counts.min() < N_FOLDS:
        raise ValueError("Not enough examples for the original 5-fold split.")
    all_data = {"layer_features": all_features, "labels": all_labels,
                "class_names": class_names}
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold, (outer_train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels_np)), labels_np), start=1
    ):
        if fold not in RUN_FOLDS:
            continue
        fit_idx, val_idx = train_test_split(
            outer_train_idx, test_size=VALIDATION_FRACTION_OF_OUTER_TRAIN,
            random_state=CV_SEED + fold, stratify=labels_np[outer_train_idx])
        assert not np.intersect1d(fit_idx, val_idx).size
        assert not np.intersect1d(fit_idx, test_idx).size
        assert not np.intersect1d(val_idx, test_idx).size
        original_train = select_rows(all_data, fit_idx)
        val_data = select_rows(all_data, val_idx)
        test_data = select_rows(all_data, test_idx)
        counts = np.bincount(original_train["labels"].numpy(), minlength=n_classes)
        if counts.min() == 0:
            raise ValueError("Missing keyword in training split.")
        for seed in FINAL_SEEDS:
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            run_start = time.perf_counter()
            smote_start = time.perf_counter()
            X_aug, y_aug = smote_features(
                original_train["layer_features"].numpy(),
                original_train["labels"].numpy(), k=5, seed=seed)
            smote_seconds = time.perf_counter() - smote_start
            counts_after = np.bincount(y_aug, minlength=n_classes)
            assert np.all(counts_after == counts.max())
            assert np.array_equal(y_aug[:len(fit_idx)], original_train["labels"].numpy())
            assert np.array_equal(X_aug[:len(fit_idx)], original_train["layer_features"].numpy())
            if not np.isfinite(X_aug).all():
                raise ValueError("Non-finite SMOTE features.")
            train_data = {"layer_features": torch.from_numpy(X_aug),
                          "labels": torch.from_numpy(y_aug).long(),
                          "class_names": class_names}
            tr, va, te = make_loaders(train_data, val_data, test_data,
                                      counts_after.tolist(), seed, oversample=False)
            # Internal CE name preserves the baseline architecture and plain CE loss.
            model, val_f1, best_epoch = train_model(
                tr, va, "CE", counts_after.tolist(), {}, seed)
            run_dir = OUTPUT_DIR / lang / "SMOTE + CE" / f"fold_{fold}" / f"seed_{seed}"
            detailed = save_run_outputs(model, va, te, counts, class_names,
                                        run_dir, val_idx, test_idx, fit_idx, counts_after)
            metrics = evaluate(model, te, counts)
            metrics.update(detailed)
            torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "method": "SMOTE + CE", "selected_config": {},
                        "use_cosine": False, "logit_scale": model.logit_scale,
                        "class_names": class_names, "train_class_counts": counts.tolist(),
                        "train_class_counts_after_smote": counts_after.tolist(),
                        "train_indices": torch.as_tensor(fit_idx),
                        "val_indices": torch.as_tensor(val_idx), "test_indices": torch.as_tensor(test_idx),
                        "fold": fold, "seed": seed, "best_epoch": best_epoch,
                        "best_validation_macro_f1": val_f1, "num_layers": all_features.shape[1],
                        "input_dim": all_features.shape[2], "dropout": DROPOUT},
                       run_dir / "best_model.pt")
            with np.load(run_dir / "outputs.npz", allow_pickle=False) as out:
                assert np.array_equal(out["train_class_counts"], counts)
                assert np.array_equal(out["train_class_counts_after_smote"], counts_after)
                assert np.array_equal(out["val_labels"], labels_np[val_idx])
                assert np.array_equal(out["test_labels"], labels_np[test_idx])
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            elapsed = time.perf_counter() - run_start
            timing = {"smote_seconds": smote_seconds, "total_run_seconds": elapsed,
                      "train_before": len(fit_idx), "train_after": len(y_aug),
                      "fold": fold, "seed": seed, "language": lang,
                      "note": "train_indices refer to real originals; synthetic rows are appended."}
            (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
            result_rows.append({"language": lang, "fold": fold, "seed": seed,
                                "method": "SMOTE + CE", "selected_config": "{}",
                                "best_validation_macro_f1": val_f1, "best_epoch": best_epoch,
                                "batch_size": BATCH_SIZE, **timing, **metrics})
            tuning_rows.append({"language": lang, "fold": fold, "seed": seed,
                                "method": "SMOTE + CE", "selected_config": "{}",
                                "smote_k": 5, "smote_target": "max", "grid_search": False,
                                "validation_macro_f1": val_f1, "best_epoch": best_epoch})
            pd.DataFrame(result_rows).to_csv(RESULTS_PATH, index=False)
            pd.DataFrame(tuning_rows).to_csv(TUNING_PATH, index=False)
            pd.DataFrame(result_rows).to_excel(OUTPUT_DIR / "classification_results.xlsx", index=False)
            pd.DataFrame(tuning_rows).to_excel(OUTPUT_DIR / "selected_hyperparameters.xlsx", index=False)
            print(f"{lang} fold={fold} seed={seed}: Macro-F1={metrics['macro_f1']:.4f}; "
                  f"SMOTE={smote_seconds:.1f}s; total={elapsed/60:.2f} min; saved {run_dir}")
            del model, tr, va, te, train_data, X_aug, y_aug
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
expected_runs = len(LANGUAGES) * len(RUN_FOLDS) * len(FINAL_SEEDS)
print(f"Completed {len(result_rows)}/{expected_runs}; total minutes="
      f"{(time.perf_counter()-experiment_start)/60:.2f}")


# %%
from IPython.display import display
results_df = pd.read_csv(RESULTS_PATH)
metric_cols = ["accuracy", "macro_f1", "micro_f1", "balanced_accuracy",
               "macro_auprc", "micro_auprc", "bottom20_precision", "bottom20_recall", "bottom20_f1"]
if results_df.duplicated(["language", "method", "fold", "seed"]).any():
    raise ValueError("Duplicate run rows")
per_language = results_df.groupby(["language", "method"])[metric_cols].agg(["mean", "std"])
per_language.to_csv(OUTPUT_DIR / "per_language_summary.csv")
fold_summary = results_df.groupby(["language", "method", "fold"])[metric_cols].agg(["mean", "std"])
fold_summary.to_csv(OUTPUT_DIR / "fold_summary.csv")
language_means = results_df.groupby(["language", "method"])[metric_cols].mean().reset_index()
overall = language_means.groupby("method")[metric_cols].agg(["mean", "std"])
overall.to_csv(OUTPUT_DIR / "overall_language_summary.csv")
expected = len(LANGUAGES) * len(METHODS) * len(RUN_FOLDS) * len(FINAL_SEEDS)
print(f"Completed {len(results_df)}/{expected} final runs. Partial summaries are provisional.")
display(per_language)
display(overall)


# %%


# %%
