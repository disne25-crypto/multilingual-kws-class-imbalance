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

OUTPUT_DIR = ROOT.parent / "All_methods_bottom20_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "classification_results_all_methods.csv"
TUNING_PATH = OUTPUT_DIR / "selected_hyperparameters_all_methods.csv"

METHODS = ["CE", "WeightedCE", "Focal", "ClassBalanced",
           "LogitAdjustment", "BalancedSoftmax", "LDAM", "RandomOversampling_CE"]
FINAL_SEEDS = [11, 22, 33]
TUNING_SEED = 2026
MAX_EPOCHS = 100
DRW_START_EPOCH = 50
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

LOW_LANGS = ["rm-sursilv", "rm-vallader", "mt", "lv", "sah", "ka", "sl", "cv", "sk", "br", "ar", "ga-IE", "ia", "fy-NL", "ta", "cnh", "zh-CN", "ro", "lt", "el", "ha"]
MEDIUM_LANGS = ["uk", "tt", "mn", "ky", "et", "cs", "pt", "sv-SE", "tr", "nl", "id", "eo"]
HIGH_LANGS = ["it", "ru", "cy", "eu", "pl"]
LANGUAGES = LOW_LANGS + MEDIUM_LANGS + HIGH_LANGS
# Confirm the original medium-resource batch size; do not guess.
BATCH_SIZE_BY_LANGUAGE = {'ar': 256, 'br': 256, 'cnh': 256, 'cv': 256, 'el': 256, 'fy-NL': 256, 'ga-IE': 256, 'ha': 256, 'ia': 256, 'ka': 256, 'lt': 256, 'lv': 256, 'mt': 256, 'rm-sursilv': 256, 'rm-vallader': 256, 'ro': 256, 'sah': 256, 'sk': 256, 'sl': 256, 'ta': 256, 'zh-CN': 256, 'mn': 2048, 'nl': 2048, 'pt': 5096, 'sv-SE': 5096, 'tr': 5096, 'tt': 2048, 'uk': 2048, 'cs': 2048, 'eo': 5096, 'et': 2048, 'id': 2048, 'ky': 2048, 'cy': 5096, 'eu': 5096, 'it': 5096, 'pl': 5096, 'ru': 5096}
# If batch sizes varied within a group, override individual language entries here.


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
    if not {"layer_features", "labels", "class_names"}.issubset(data):
        raise ValueError(f"{path} is not from the updated feature notebook")
    return data


# %%
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Normalise each feature vector and each class weight vector.
        x = nn.functional.normalize(x, p=2, dim=1)
        weight = nn.functional.normalize(self.weight, p=2, dim=1)

        # Return cosine scores; LDAM loss applies the scale of 30.
        return nn.functional.linear(x, weight)


class WAPMLP(nn.Module):
    def __init__(
        self, num_layers, input_dim, num_classes,
        use_cosine=False, logit_scale=1.0
    ):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.logit_scale = logit_scale

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
        layer_weights = torch.softmax(self.layer_logits, dim=0)
        pooled = (x * layer_weights[None, :, None]).sum(dim=1)
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
    prob_parts, label_parts = [], []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        prob_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
        label_parts.append(y.numpy())

    probs = np.concatenate(prob_parts)
    y_true = np.concatenate(label_parts)
    y_pred = probs.argmax(axis=1)
    class_ids = np.arange(len(counts))
    recalls = []
    for class_id in class_ids:
        mask = y_true == class_id
        recalls.append(float((y_pred[mask] == class_id).mean()) if mask.any() else 0.0)
    rare_n = max(1, int(math.ceil(0.20 * len(class_ids))))
    rare_ids = np.argsort(np.asarray(counts))[:rare_n]
    y_bin = label_binarize(y_true, classes=class_ids)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=class_ids,
                             average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, labels=class_ids,
                             average="micro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_auprc": average_precision_score(y_bin, probs, average="macro"),
        "micro_auprc": average_precision_score(y_bin, probs, average="micro"),
        "bottom20_recall": float(np.mean(np.asarray(recalls)[rare_ids])),
    }


def train_model(train_loader, val_loader, method, counts, cfg, seed):
    set_seed(seed)
    example_x, _ = next(iter(train_loader))
   

    model = WAPMLP(
        num_layers=example_x.shape[1],
        input_dim=example_x.shape[2],
        num_classes=len(counts),
        use_cosine=(method == "LDAM"),
        logit_scale=cfg.get("scale", 30.0) if method == "LDAM" else 1.0,
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_score, best_state, best_epoch, wait = -1.0, None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(method, model(x), y, counts, cfg, epoch)
            loss.backward()
            optimizer.step()
        val_score = evaluate(model, val_loader, counts)["macro_f1"]
        print(f"{method} epoch {epoch}: validation Macro-F1={val_score:.4f}")
        if val_score > best_score:
            best_score, best_epoch, wait = val_score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            if method != "LDAM" or not cfg.get("drw", False) or epoch >= DRW_START_EPOCH:
                wait += 1
            else:
                wait = 0
        if wait >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")
    model.load_state_dict(best_state)
    return model, best_score, best_epoch


# %%
def bottom20_details(y_true, y_pred, counts):
    class_ids = np.arange(len(counts))
    # Match the original rare-keyword selection, including its tie behaviour.
    rare_ids = np.argsort(np.asarray(counts))[:max(1, int(math.ceil(0.2 * len(counts))))]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    return rare_ids, precision, recall, f1, support, cm


@torch.inference_mode()
def save_run_outputs(model, va, te, counts, class_names, run_dir,
                     val_idx, test_idx, fit_idx):
    model.eval()
    def collect(loader):
        logits_parts, label_parts = [], []
        for x, y in loader:
            logits_parts.append(model(x.to(DEVICE)).cpu().numpy())
            label_parts.append(y.numpy())
        return np.concatenate(logits_parts), np.concatenate(label_parts)
    val_logits, val_labels = collect(va)
    test_logits, test_labels = collect(te)
    # Preserve the original softmax-argmax evaluation rule.
    pred = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy().argmax(axis=1)
    rare_ids, precision, recall, f1, support, cm = bottom20_details(test_labels, pred, counts)
    names = np.asarray(class_names, dtype=str)
    np.savez(run_dir / "outputs.npz",
             val_logits=val_logits, val_labels=val_labels,
             test_logits=test_logits, test_labels=test_labels,
             test_predictions=pred, train_class_counts=np.asarray(counts),
             class_names=names, bottom20_class_ids=rare_ids,
             confusion_matrix=cm, bottom20_confusion_rows=cm[rare_ids, :],
             bottom20_confusion_columns=cm[:, rare_ids],
             train_indices=fit_idx, val_indices=val_idx, test_indices=test_idx)
    pd.DataFrame({"class_id":np.arange(len(counts)), "keyword":names,
                  "train_count":counts, "test_support":support,
                  "precision":precision, "recall":recall, "f1":f1,
                  "is_bottom20":np.isin(np.arange(len(counts)), rare_ids)}).to_csv(
                      run_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame({"class_id":rare_ids, "keyword":names[rare_ids],
                  "train_count":np.asarray(counts)[rare_ids], "test_support":support[rare_ids],
                  "precision":precision[rare_ids], "recall":recall[rare_ids],
                  "f1":f1[rare_ids]}).to_csv(run_dir / "bottom20_metrics.csv", index=False)
    pd.DataFrame(cm, index=names, columns=names).rename_axis("true_keyword").to_csv(
        run_dir / "confusion_matrix_full.csv")
    pd.DataFrame(cm[rare_ids, :], index=names[rare_ids], columns=names).rename_axis(
        "true_keyword").to_csv(run_dir / "bottom20_confusion_rows.csv")
    pd.DataFrame(cm[:, rare_ids], index=names, columns=names[rare_ids]).rename_axis(
        "true_keyword").to_csv(run_dir / "bottom20_confusion_columns.csv")
    return {"bottom20_precision":float(precision[rare_ids].mean()),
            "bottom20_recall":float(recall[rare_ids].mean()),
            "bottom20_f1":float(f1[rare_ids].mean())}


# %%
if not any(p.is_dir() for p in ROOT.iterdir()):
    raise RuntimeError(f"No language folders found in {ROOT}")

N_FOLDS = 5
CV_SEED = 42
VALIDATION_FRACTION_OF_OUTER_TRAIN = 0.125  # 10% of all data when outer test fold is 20%
result_rows, tuning_rows = [], []

language_dirs = [ROOT / code for code in LANGUAGES]
for lang in LANGUAGES:
    value = BATCH_SIZE_BY_LANGUAGE.get(lang)
    if value is None or value <= 0:
        raise ValueError(f"Set the original batch size for {lang} in the settings cell.")

for lang_dir in language_dirs:
    for split in ("train", "val", "test"):
        path = lang_dir / f"{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature file: {path}")

print("Running all methods for:", [p.name for p in language_dirs])
print("Number of languages:", len(language_dirs))

def select_rows(data, indices):
    indices = torch.as_tensor(indices, dtype=torch.long)
    return {
        "layer_features": data["layer_features"].index_select(0, indices),
        "labels": data["labels"].index_select(0, indices),
        "class_names": data["class_names"],
    }


for lang_dir in language_dirs:
    lang = lang_dir.name
    BATCH_SIZE = BATCH_SIZE_BY_LANGUAGE[lang]
    saved_splits = [load_split(lang_dir, name) for name in ("train", "val", "test")]
    class_names = saved_splits[0]["class_names"]
    if any(part["class_names"] != class_names for part in saved_splits[1:]):
        raise ValueError(f"Class mapping differs across saved files: {lang}")

    # The earlier train/val/test folders contain extracted features only.
    # Recombine them here so every recording is held out in exactly one CV fold.
    all_features = torch.cat([part["layer_features"] for part in saved_splits], dim=0)
    all_labels = torch.cat([part["labels"] for part in saved_splits], dim=0)
    labels_np = all_labels.numpy()
    all_data = {
        "layer_features": all_features,
        "labels": all_labels,
        "class_names": class_names,
    }
    n_classes = len(class_names)
    full_counts = np.bincount(labels_np, minlength=n_classes)
    if len(full_counts) != n_classes or full_counts.min() < N_FOLDS:
        raise ValueError(
            f"{lang}: 5-fold stratification needs at least 5 examples per class; "
            f"minimum is {full_counts.min()}"
        )

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    print(f"\n{lang}: {len(class_names)} classes; {len(all_labels)} recordings")

    for fold, (outer_train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels_np)), labels_np), start=1
    ):
        # Test fold stays untouched. Split the other 80% into train and validation.
        fit_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=VALIDATION_FRACTION_OF_OUTER_TRAIN,
            random_state=CV_SEED + fold,
            stratify=labels_np[outer_train_idx],
        )
        train_data = select_rows(all_data, fit_idx)
        val_data = select_rows(all_data, val_idx)
        test_data = select_rows(all_data, test_idx)

        train_count_map = Counter(train_data["labels"].tolist())
        counts = [train_count_map.get(i, 0) for i in range(n_classes)]
        if min(counts) == 0:
            raise ValueError(f"A class is absent from fold {fold} training data for {lang}")
        print(
            f" Fold {fold}/{N_FOLDS}: train={len(fit_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )

        for method in METHODS:
            candidates = TUNING_GRIDS.get(method, [{}])
            best_cfg, best_val, best_epoch = {}, -1.0, 0
            for cfg in candidates:
                tr, va, _ = make_loaders(
                    train_data, val_data, test_data, counts, TUNING_SEED + fold,
                    oversample=(method == "RandomOversampling_CE"),
                )
                _, score, epoch = train_model(
                    tr, va, method, counts, cfg, TUNING_SEED + fold
                )
                print(f"  tune {method} {cfg}: validation Macro-F1={score:.4f}")
                if score > best_val:
                    best_cfg, best_val, best_epoch = cfg, score, epoch

            tuning_rows.append({
                "language": lang, "fold": fold, "method": method,
                "selected_config": json.dumps(best_cfg, sort_keys=True),
                "validation_macro_f1": best_val, "best_epoch": best_epoch,
            })

            for seed in FINAL_SEEDS:
                tr, va, te = make_loaders(
                    train_data, val_data, test_data, counts, seed,
                    oversample=(method == "RandomOversampling_CE"),
                )
                
                model, val_f1, best_epoch_seed = train_model(
                    tr, va, method, counts, best_cfg, seed
                )
                
                run_dir = OUTPUT_DIR / lang / method / f"fold_{fold}" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                detailed_metrics = save_run_outputs(
                    model, va, te, counts, class_names, run_dir,
                    val_idx, test_idx, fit_idx)

                # Save the trained model and exact split indices too.
                torch.save(
                    {
                        "model_state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "class_names": class_names,
                        "train_class_counts": counts,
                        "train_indices": torch.as_tensor(fit_idx),
                        "val_indices": torch.as_tensor(val_idx),
                        "test_indices": torch.as_tensor(test_idx),
                        "fold": fold,
                        "seed": seed,
                        "best_epoch": best_epoch_seed,
                        "best_validation_macro_f1": val_f1,
                        "num_layers": all_features.shape[1],
                        "input_dim": all_features.shape[2],
                        "dropout": DROPOUT,
                    },
                    run_dir / "best_model.pt",
                )
                
                test_metrics = evaluate(model, te, counts)
                test_metrics.update(detailed_metrics)
   
                
                result_rows.append({
                    "language": lang, "fold": fold, "method": method, "seed": seed,
                    "selected_config": json.dumps(best_cfg, sort_keys=True),
                    "best_validation_macro_f1": val_f1,
                    "best_epoch": best_epoch_seed,
                    "batch_size": BATCH_SIZE,
                    **test_metrics,
                })
                # Save after every run so completed work remains available if interrupted.
                pd.DataFrame(result_rows).to_csv(RESULTS_PATH, index=False)
                pd.DataFrame(tuning_rows).to_csv(TUNING_PATH, index=False)
                print(
                    f"  TEST {method}, seed={seed}: "
                    f"Macro-F1={test_metrics['macro_f1']:.4f}"
                )

print("Saved fold/seed results:", RESULTS_PATH)
print("Saved fold-specific tuning settings:", TUNING_PATH)


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
expected = len(LANGUAGES) * len(METHODS) * N_FOLDS * len(FINAL_SEEDS)
print(f"Completed {len(results_df)}/{expected} final runs. Partial summaries are provisional.")
display(per_language)
display(overall)


# %%


# %%
