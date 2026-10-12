# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import random
import numpy as np
import torch
import torchaudio
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)


# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# %%
DATA_ROOTS = {
    "low": Path('./data/low_resource'),
    "medium": Path('./data/medium_resource'),
    "high": Path('./data/high_resource'),
}

# Exactly the ROOT used by All_38_languages_8_methods(1).ipynb.
FEATURE_ROOT = Path(
    './features'
)

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
TARGET_SR = 16000
MIN_PER_KEYWORD = 15
TOP_K = 20

LOW_LANGS = ['rm-sursilv', 'rm-vallader', 'mt', 'lv', 'sah', 'ka', 'sl', 'cv', 'sk', 'br', 'ar', 'ga-IE', 'ia', 'fy-NL', 'ta', 'cnh', 'zh-CN', 'ro', 'lt', 'el', 'ha']
MEDIUM_LANGS = ['uk', 'tt', 'mn', 'ky', 'et', 'cs', 'pt', 'sv-SE', 'tr', 'nl', 'id', 'eo']
HIGH_LANGS = ['it', 'ru', 'cy', 'eu', 'pl']
LANGUAGES_BY_GROUP = {
    "low": LOW_LANGS,
    "medium": MEDIUM_LANGS,
    "high": HIGH_LANGS,
}
LANGUAGES = LOW_LANGS + MEDIUM_LANGS + HIGH_LANGS
assert len(LANGUAGES) == len(set(LANGUAGES)) == 38

# Check all source folders before starting expensive extraction.
for group, root in DATA_ROOTS.items():
    if not root.is_dir():
        raise FileNotFoundError(f"{group} data path not found: {root}")
    for lang in LANGUAGES_BY_GROUP[group]:
        if not (root / lang).is_dir():
            raise FileNotFoundError(f"Missing {group} language folder: {root / lang}")

FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
print("Languages:", len(LANGUAGES))
print("Feature output:", FEATURE_ROOT)


# %%
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
encoder = Wav2Vec2Model.from_pretrained(
    MODEL_NAME, output_hidden_states=True
).to(DEVICE)
encoder.eval()
for parameter in encoder.parameters():
    parameter.requires_grad_(False)

print("Hidden states:", encoder.config.num_hidden_layers + 1)
print("Hidden dimension:", encoder.config.hidden_size)


# %%
def load_waveform(path):
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SR
        )
    waveform = waveform.squeeze(0).contiguous().float()
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio: {path}")
    return waveform


@torch.inference_mode()
def extract_layer_means(path):
    waveform = load_waveform(path)
    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).input_values.to(DEVICE)
    outputs = encoder(input_values=inputs, output_hidden_states=True)
    # One mean vector per hidden layer; stack into [layers, hidden_dimension].
    layer_means = torch.stack(
        [hidden[0].mean(dim=0) for hidden in outputs.hidden_states], dim=0
    )
    return layer_means.cpu().to(torch.float16)


# %%
def make_splits(paths, labels, seed=SEED):
    indices = np.arange(len(paths))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.20, random_state=seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed,
        stratify=np.asarray(labels)[temp_idx]
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


# %%
summary = []

for resource_group, data_root in DATA_ROOTS.items():
    language_dirs = [data_root / lang for lang in LANGUAGES_BY_GROUP[resource_group]]
    print(f"\\n{resource_group}: {len(language_dirs)} language folders")

    for lang_dir in language_dirs:
        keyword_to_files = {}
        for kw_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            wavs = sorted(kw_dir.glob("*.wav"))
            if len(wavs) >= MIN_PER_KEYWORD:
                keyword_to_files[kw_dir.name] = wavs

        ranked = sorted(
            keyword_to_files.items(),
            key=lambda item: (-len(item[1]), item[0])
        )[:TOP_K]
        if len(ranked) < TOP_K:
            print(f"SKIP {lang_dir.name}: only {len(ranked)} eligible keywords")
            continue

        class_names = [kw for kw, _ in ranked]
        paths, labels = [], []
        for label, (kw, wavs) in enumerate(ranked):
            paths.extend(wavs)
            labels.extend([label] * len(wavs))

        splits = make_splits(paths, labels)
        lang_out = FEATURE_ROOT / lang_dir.name
        lang_out.mkdir(parents=True, exist_ok=True)
        print(f"\\n{lang_dir.name}: {len(paths)} clips, {len(class_names)} classes")

        for split_name, indices in splits.items():
            layer_rows = []
            for idx in tqdm(indices, desc=f"{lang_dir.name}/{split_name}"):
                layer_rows.append(extract_layer_means(paths[int(idx)]))
            saved = {
                "layer_features": torch.stack(layer_rows),
                "labels": torch.tensor(
                    [labels[int(i)] for i in indices], dtype=torch.long
                ),
                "class_names": class_names,
                "file_paths": [str(paths[int(i)]) for i in indices],
                "resource_group": resource_group,
                "model_name": MODEL_NAME,
                "sample_rate": TARGET_SR,
            }
            torch.save(saved, lang_out / f"{split_name}.pt")
            print(f"Saved {split_name}: {len(indices)} clips")

        train_labels = [labels[int(i)] for i in splits["train"]]
        counts = np.bincount(train_labels, minlength=len(class_names))
        summary.append({
            "language": lang_dir.name,
            "resource_group": resource_group,
            "keywords": len(class_names),
            "clips": len(paths),
            "train": len(splits["train"]),
            "validation": len(splits["val"]),
            "test": len(splits["test"]),
            "train_min_class": int(counts.min()),
            "train_max_class": int(counts.max()),
            "train_IR": float(counts.max() / counts.min()),
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(FEATURE_ROOT / "dataset_summary.csv", index=False)
display(summary_df)


# %%
missing = [
    str(FEATURE_ROOT / lang / f"{split}.pt")
    for lang in LANGUAGES
    for split in ("train", "val", "test")
    if not (FEATURE_ROOT / lang / f"{split}.pt").is_file()
]
if missing:
    raise RuntimeError("Missing feature files:\n" + "\n".join(missing))
print("All 38 languages have train.pt, val.pt and test.pt.")
