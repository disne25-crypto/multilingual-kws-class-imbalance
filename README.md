# Anonymous multilingual keyword-spotting code

## Files and execution order
1. `01_feature_extraction.ipynb`: prepare 38-language top-20 keyword datasets and frozen XLS-R-300M layer-wise means.
2. `02_main_classification.ipynb`: eight methods, five-fold stratified CV and final seeds 11, 22, 33. Includes cosine-LDAM and validation-only selection.
3. `03_smote.ipynb`: training-only feature-space SMOTE followed by CE, five folds and three seeds.
4. `04_x1.ipynb`: long-tail and matched-scarcity arms, CE, Weighted CE, Class-Balanced and ROS + CE; five folds and final seed 11. Batch size scales with subset size.

Equivalent `.py` files use `# %%` cell separators. Notebook display calls are intended for an IPython environment.

## Setup
Install dependencies with `pip install -r requirements.txt`. The requirements list is a dependency list, not a lockfile of the original training environment. For exact reproduction use the original environment versions.
Run from this directory, use a fresh output folder for each run, and select an available GPU in the first cell (default cuda:0, otherwise CPU).

## Data layout
Obtain the Multilingual Spoken Words Corpus separately. Arrange WAV files as `data/{low_resource,medium_resource,high_resource}/<language>/<keyword>/*.wav`. Edit DATA_ROOTS if needed. Extraction produces `features/<language>/{train,val,test}.pt` with layer_features, labels and class_names.
Do not run extraction over existing experiment features: use a new feature destination. Exact CV reproduction depends on retaining the same recording order and selected dataset version.

## Experiment settings
The main notebook recombines saved train/val/test features in that order, applies five-fold stratification (seed 42), and takes validation from outer training with fraction 0.125 and seed 42+fold. Maximum epochs 100, patience 10, Adam learning rate 0.001, dropout 0.30; select by validation Macro-F1.
Main batch sizes: low languages 256; mn,nl,tt,uk,cs,et,id,ky 2048; pt,sv-SE,tr,eo,cy,eu,it,pl,ru 5096, verified against the supplied batch-size spreadsheet. See `batch_sizes.csv` for all 38 values.

SMOTE and X1 default to all 21 low-resource languages. All 38 language codes are included; set LANGUAGES = ALL_LANGUAGES for an optional extension, which was not part of the reported low-resource experiments. Give independent jobs different OUTPUT_DIR values. X1 uses seed 11 only; retain the subset seed and scaled batch formula.

## Outputs
Per-run NPZ files include logits, labels, class counts and fold indices; CSVs include per-class and bottom-20 metrics and confusion matrices. SMOTE stores real counts plus augmented counts; real counts define rare keywords. X1 stores subset indices and achieved IR (which can be below the target).

## Scope of this package
This archive contains code only. It does not include audio, cached features, trained models, actual result tables, or original recording-level split manifests. Add the actual manifests and reported result CSVs separately when available. Generated index arrays alone need the corresponding recording-order mapping. Proportional SMOTE and the alternative IR50 design are not included.
