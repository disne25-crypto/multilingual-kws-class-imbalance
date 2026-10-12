# Class Imbalance Handling in Multilingual Keyword Spotting

## Overview

This repository provides supplementary experimental code for a
systematic investigation of class imbalance handling strategies
in multilingual isolated keyword spotting (KWS).

The experiments cover 38 languages from the Multilingual Spoken
Words Corpus (MSWC), using frozen XLS-R-300M speech representations.

The study evaluates whether imbalance-aware learning strategies
improve overall classification performance and recognition of
underrepresented keywords.

The repository includes the main classification benchmark,
feature-space SMOTE experiments, and controlled long-tail and
matched-scarcity experiments.

## Repository Structure

The files are organized according to the experimental workflow.

| File | Description |
|------|-------------|
| `01_feature_extraction.ipynb` | Dataset preparation and frozen XLS-R-300M feature extraction |
| `02_main_classification.ipynb` | Main benchmark evaluating eight imbalance-handling methods |
| `03_smote.ipynb` | Feature-space SMOTE experiments |
| `04_x1.ipynb` | Long-tail and matched-scarcity experiments |
| `requirements.txt` | Python dependencies |
| `batch_sizes.csv` | Batch-size configurations for the 38 languages |

Equivalent Python scripts are provided using `# %%` cell
separators.

Notebook display functions are intended for execution in
an IPython or Jupyter environment.

## Dataset

The experiments use the Multilingual Spoken Words Corpus (MSWC).

A total of 38 languages are considered.

Languages are grouped according to resource availability:

- Low-resource: 21 languages
- Medium-resource: 12 languages
- High-resource: 5 languages

For each language, the top 20 selected keywords are used.

Each selected keyword satisfies the minimum recording-count
criterion of 15 recordings.

### Language Groups

**Low-resource languages (21)**

ta, rm-sursilv, rm-vallader, mt, lv, sah, ka, sl, cv,
sk, br, ar, ga-IE, fy-NL, ia, cnh, zh-CN, ro, lt,
el, ha

**Medium-resource languages (12)**

uk, tt, mn, ky, et, cs, pt, sv-SE, tr, nl, id, eo

**High-resource languages (5)**

pl, ru, it, cy, eu

## Feature Extraction

Speech representations are extracted using the pretrained
XLS-R-300M model.

The feature extractor remains frozen throughout the
classification experiments.

Audio recordings are processed at a sampling rate of 16 kHz.

Layer-wise mean representations are extracted from the
pretrained model and saved for subsequent classification.

The feature extraction notebook prepares the selected
keyword datasets and saves the corresponding representations.

### Feature Directory Structure

The expected input audio directory structure is:

```text
data/
    low_resource/
        <language>/
            <keyword>/
                *.wav

    medium_resource/
        <language>/
            <keyword>/
                *.wav

    high_resource/
        <language>/
            <keyword>/
                *.wav
```

Extracted features are stored as:

```text
features/
    <language>/
        train.pt
        val.pt
        test.pt
```

Feature files contain layer-wise representations, labels,
and class names.

The classification notebook recombines the saved feature
partitions before applying cross-validation.

For exact reproduction, the original recording order,
dataset version, and partition assignments must be retained.

## Main Classification Benchmark

The main experiment evaluates eight class imbalance
handling strategies.

### Methods

1. Cross-Entropy (CE)
2. Weighted Cross-Entropy
3. Focal Loss
4. Class-Balanced Loss
5. Logit Adjustment
6. Balanced Softmax
7. Random Oversampling with Cross-Entropy (ROS + CE)
8. Label-Distribution-Aware Margin Loss (LDAM)

All methods use the same frozen speech representations
and a common classification architecture.

### Classification Architecture

The classifier consists of:

- Learnable layer-weighted pooling
- Two-layer multilayer perceptron (MLP)
- Dropout regularization
- Keyword classification output layer

Layer weights are learned using a softmax-normalized
weighting mechanism.

## Cross-Validation Protocol

The main experiments use stratified five-fold
cross-validation.

The same data partitions are used across methods
and random seeds.

### Cross-Validation Settings

- Number of folds: 5
- Cross-validation seed: 42
- Validation fraction: 0.125 of outer training data
- Validation seed: 42 + fold
- Final training seeds: 11, 22, 33

For each outer fold:

1. Approximately 20% of recordings are reserved for testing.
2. The remaining 80% form the outer training partition.
3. A validation subset is selected from the outer training data.
4. Hyperparameters are selected using validation performance.
5. The best-validation checkpoint is evaluated on the
   held-out test partition.

The test partition is not used for hyperparameter selection.

## Training Configuration

The main training configuration is:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Dropout | 0.30 |
| Model selection metric | Validation Macro-F1 |
| Number of folds | 5 |
| Final random seeds | 11, 22, 33 |

### Batch Sizes

Batch sizes vary according to the language and
experimental configuration.

**Low-resource languages**

All 21 low-resource languages use batch size 256.

**Batch size 2048**

mn, nl, tt, uk, cs, et, id, ky

**Batch size 5096**

pt, sv-SE, tr, eo, cy, eu, it, pl, ru

The complete language-specific batch-size configuration
is provided in `batch_sizes.csv`.

## Imbalance-Handling Methods

### Cross-Entropy

Standard cross-entropy serves as the baseline method.

### Weighted Cross-Entropy

Class weights are calculated using inverse
class frequencies.

### Focal Loss

The focusing parameter gamma is selected from:

- 1
- 2
- 3

The alpha class-weighting term is not used.

### Class-Balanced Loss

Class-Balanced Loss uses effective-number-based
class weighting.

The beta parameter is selected from:

- 0.99
- 0.999
- 0.9999

### Logit Adjustment

Class-frequency information is incorporated through
logit adjustment.

### Balanced Softmax

Balanced Softmax accounts for class-frequency
differences during classification.

### Random Oversampling

Random oversampling is applied only to the training
partition.

Validation and test partitions retain their original
class distributions.

### LDAM

LDAM experiments use a cosine-normalized
classification layer.

The final classifier normalizes feature vectors
and classifier weights before applying a scale
factor of 30.

The evaluated margin parameters are:

- 0.25
- 0.5

Experiments consider LDAM with and without
deferred reweighting (DRW).

For DRW configurations, the early-stopping counter
starts after epoch 50.

Hyperparameters are selected using validation
Macro-F1.

## SMOTE Experiments

The SMOTE experiments investigate synthetic
oversampling in the extracted feature space.

SMOTE is applied exclusively to the training
partition.

The validation and test partitions remain unchanged.

The experiments use:

- Five-fold stratified cross-validation
- Three random seeds: 11, 22, 33
- Cross-Entropy classification
- Training-only feature-space SMOTE

The provided SMOTE notebook defaults to the
21 low-resource languages.

The code includes all 38 language identifiers,
allowing optional extension to other resource groups.

Such extensions are not part of the reported
low-resource SMOTE experiments.

Both original training class counts and
augmented training class counts are retained.

Minority classes are identified using the
original training distribution.

## X1: Long-Tail and Matched-Scarcity Experiments

The X1 experiments investigate the effects of
class imbalance and data scarcity under
controlled experimental conditions.

Two experimental settings are considered:

1. Long-tail class distributions
2. Matched-scarcity configurations

The experiments evaluate four methods:

- Cross-Entropy
- Weighted Cross-Entropy
- Class-Balanced Loss
- Random Oversampling with Cross-Entropy

### Experimental Configuration

- Five-fold stratified cross-validation
- Three random seeds: 11, 22, 33
- Validation-based model selection
- Training-subset sampling
- Batch size adjusted according to subset size

The X1 notebook defaults to all 21
low-resource languages.

The code also contains all 38 language identifiers
for optional extensions.

The generated experimental outputs retain
the selected training-subset indices.

The achieved imbalance ratio is recorded because
the realized ratio may differ from the target
under sampling constraints.

The proportional SMOTE configuration and
alternative IR50 design are not included
in this package.

## Evaluation Metrics

The main experiments evaluate classification
performance using:

1. Accuracy
2. Macro-F1
3. Micro-F1
4. Balanced Accuracy
5. Macro-AUPRC
6. Micro-AUPRC
7. Bottom-20% keyword recall

Additional analyses consider:

- Bottom-20% keyword precision
- Per-class classification performance
- Confusion matrices

### Minority Keyword Definition

Minority keywords are defined as the least
frequent 20% of keyword classes according
to the training-set class distribution.

For experiments with 20 keyword classes,
the four least frequent classes are selected.

The same training-frequency-based definition
is used when evaluating minority-keyword
performance.

## Experimental Outputs

The experimental scripts support saving
per-run predictions and evaluation outputs.

Depending on the experiment, generated files include:

- Validation logits
- Validation labels
- Test logits
- Test labels
- Training class counts
- Fold indices
- Per-class evaluation metrics
- Bottom-20% evaluation metrics
- Confusion matrices
- Experiment configuration information

Results are organized by language,
cross-validation fold, and random seed.

Summary CSV files support aggregation
across folds, seeds, and languages.

The SMOTE experiments additionally retain
original and augmented training counts.

The X1 experiments retain sampled subset
indices and achieved imbalance ratios.

These files are generated when the
corresponding experiments are executed.

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The requirements file provides a dependency list
rather than an exact lockfile of the original
training environment.

Exact environment reproduction requires
the original package and framework versions.

A CUDA-enabled GPU is recommended.

## Execution Order

Run the experiments in the following order.

### Step 1: Feature Extraction

```text
01_feature_extraction.ipynb
```

Prepare the MSWC recordings and extract frozen
XLS-R-300M representations.

### Step 2: Main Classification

```text
02_main_classification.ipynb
```

Run the eight imbalance-handling methods using
five-fold cross-validation and three random seeds.

### Step 3: SMOTE

```text
03_smote.ipynb
```

Run training-only feature-space SMOTE experiments.

### Step 4: X1

```text
04_x1.ipynb
```

Run the long-tail and matched-scarcity experiments.

### Execution Notes

- Update dataset paths before execution.
- Select an available GPU in the configuration cell.
- The default device is cuda:0 when available.
- CPU execution is supported but may be slow.
- Use a separate output directory for each experiment.
- Do not overwrite previously extracted features.
- Do not execute multiple jobs using the same output directory.

## Reproducibility Notes

The experiments are designed to support controlled
comparisons across imbalance-handling methods.

The main benchmark uses common data partitions,
classifier architecture, and evaluation procedures.

Hyperparameters are selected using validation data.

Test partitions are reserved for final evaluation.

For exact reproduction, the original dataset version,
recording order, partition assignments, and
software environment must be preserved.

## Scope and Limitations

This repository contains experimental code and
configuration information.

It does not include:

- Original MSWC audio recordings
- Cached extracted speech representations
- Trained model checkpoints
- Original experimental result tables
- Original recording-level split manifests

The MSWC dataset must be obtained separately.

Generated fold indices require the corresponding
recording-order mapping for exact reproduction.

The repository provides the experimental implementation,
but complete numerical reproduction depends on
the original data ordering and environment.

## Anonymous Supplementary Material

This repository is provided as supplementary material
for anonymous peer review.

Identifying information is intentionally excluded
from the supplementary documentation.
