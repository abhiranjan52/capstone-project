# Multimodal Weight-Regression Training Pipeline

Builds on the preprocessed `processed_dataset/` produced by
`build_dataset.py`. Trains two unimodal regressors (CSI-only, video-only)
to convergence separately, then fuses them with a simple feature-
concatenation baseline, and evaluates all three on a held-out test set —
overall, and split by whether the sample was visually occluded.

Requires `torch` and `torchvision`; if the pretrained ResNet18 weights
can't be downloaded (e.g. no internet access, blocked registry), the
video backbone automatically falls back to a small from-scratch CNN with
a printed warning, so the pipeline still runs either way.

## Run order
```bash
pip install torch torchvision scikit-learn pandas numpy scipy

python splits.py         # 1. build the train/val/test split (occlusion-stratified)
python fit_scalers.py    # 2. fit WeightScaler + CSIScaler on the TRAIN split only
python train_csi.py      # 3. train CSI-only regressor to convergence
python train_video.py    # 4. train video-only regressor to convergence
python train_fusion.py   # 5. baseline fusion: concat CSI+video features, train new head
python evaluate.py       # 6. compare all three, overall + occluded vs non-occluded
```
`fit_scalers.py` is technically optional — `train_csi.py`/`train_fusion.py`
fit and save both scalers automatically if missing — but running it
explicitly makes the normalization step visible rather than implicit
inside a training script, and guarantees both scalers exist before
`evaluate.py` (which only loads, never fits, to avoid silently refitting
on a different split by accident).

Each stage is independent once its checkpoint exists — `train_fusion.py`
or `evaluate.py` can be rerun without repeating earlier steps.

## Files
- **`config_train.py`** — all training hyperparameters, kept separate
  from `config.py` (the preprocessing config) so nothing here touches
  preprocessing settings.
- **`splits.py`** — builds the train/val/test split, **stratified by
  `(weight_g, occluded)`** so every split gets a realistic mix of occluded
  and non-occluded samples. `occluded = Visual_Occlusion > 0`, used only
  for stratifying/reporting — never fed to any model as an input.
  - **Caveat**: a ledger with only a handful of sessions (and few with an
    occlusion variant) is too small for a clean *session*-level holdout
    stratified by weight+occlusion — most strata would have exactly one
    session. The default `SPLIT_STRATEGY="trial"` splits at the
    individual-trial level instead, which risks same-session trials
    leaking across train/test (trials from one session likely share
    lighting/background/setup). `SPLIT_STRATEGY="session"` is implemented
    and activates automatically once there are enough sessions per
    stratum — until then it falls back to trial-level with a printed
    warning.
- **`train_datasets.py`** — three `Dataset` classes (`CSIOnlyDataset`,
  `VideoOnlyDataset`, `MultimodalDataset`) plus two scalers, both fit on
  **train-split statistics only** (no leakage):
  - `WeightScaler` — standardizes the `weight_g` target, saved to
    `processed_dataset/target_scaler.json`, used to invert predictions
    back to real grams for every reported metric.
  - `CSIScaler` — z-score normalizes CSI amplitude and phase separately
    (global mean/std across all receivers/time/subcarriers in the train
    split), saved to `processed_dataset/csi_scaler.json`. **Mask-aware**:
    zero-filled entries from a missing receiver (see `csi.py`'s
    `extract_trial_csi`) are excluded from the fitted mean/std and stay
    exactly zero after transform, so they remain a clean "missing" signal
    rather than being shifted to some arbitrary normalized value. Fit by
    streaming through train `.npz` files one at a time rather than
    loading the whole split into memory.

  Normalization lives entirely in this training pipeline, not in
  `build_dataset.py` — `build_dataset.py` saves raw CSI (filtered by the
  Hampel/lowpass/DC pipeline in `csi.py`, but unnormalized), and
  normalization is fit here, after the split exists, so no test-set
  values leak into it.

  Video frames are sub-sampled per training step from each clip's valid
  (non-padded) range — random frames for train (cheap temporal
  augmentation), evenly-spaced for val/test (reproducible).
- **`fit_scalers.py`** — explicit step that fits and saves both scalers
  from the train split. Run after `splits.py`, before training.
- **`models.py`**:
  - `CSIEncoder` — small CNN2D over the `(n_rx*2, T, subcarriers)` CSI
    tensor (receivers × [amplitude,phase] as channels), global-pooled to a
    feature vector.
  - `VideoClipEncoder` — per-frame CNN backbone (pretrained ResNet18 if
    available, else a small from-scratch CNN) + masked-mean temporal
    pooling, run separately per camera then concatenated.
  - `CSIRegressor` / `VideoRegressor` — encoder + small MLP head, for the
    standalone stage-1/2 models.
  - `FusionConcatRegressor` — the baseline fusion approach: takes the
    already-trained `CSIEncoder` and `VideoClipEncoder`, concatenates
    their feature vectors, and trains a new head on top. No cross-modal
    attention or gating — that's left for a later iteration on top of
    this baseline.
- **`train_utils.py`** — shared `EarlyStopping`, regression metrics
  (MAE/RMSE/R² in real grams, not the standardized scale), and generic
  train/eval epoch loops reused by all three `train_*.py` scripts.
- **`train_csi.py`, `train_video.py`, `train_fusion.py`** — one script per
  stage, each trains to convergence (early stopping on val loss, patience
  configurable), saves the best checkpoint, reports final test metrics,
  and prints one line per epoch (train loss, val loss, val MAE, val R²,
  whether it's a new best) so training progress is visible as it runs.
- **`evaluate.py`** — loads whichever checkpoints exist, evaluates each on
  the test set **overall, non-occluded only, and occluded only**, saves
  `results/comparison.csv`, and prints a headline video-vs-fusion delta.

## Design decisions (change in `config_train.py` if these don't fit)
- **Loss**: Huber/SmoothL1 (robust to outlier trials) rather than plain
  MSE; metrics are reported in MAE/RMSE/R² in real grams for
  interpretability regardless of the training loss.
- **Fusion baseline = frozen backbones + new head**
  (`FREEZE_BACKBONES_IN_FUSION = True` by default). This isolates
  "does concatenating CSI features onto an already-converged video model
  help" as cleanly as possible, and is the least likely to overfit on a
  small dataset. Setting it to `False` jointly fine-tunes both encoders
  with the new head at a reduced LR (`FUSION_BACKBONE_LR_MULT`) instead —
  more capacity, more overfitting risk on a small dataset.
- **Occlusion is never a model input** — it only drives split
  stratification (`splits.py`) and test-time slicing (`evaluate.py`).
  Training batches mix occluded and non-occluded samples freely and the
  model has no way to know which is which.
- **Pretrained video backbone**: ResNet18 (ImageNet) if `torchvision` can
  download it; otherwise a small CNN trained from scratch, with a printed
  warning either way. Recommended to keep pretrained given how little
  video data is typically available for this kind of dataset.

## Normalization and split-leakage handling
CSI normalization statistics are computed exclusively from the train
split, after the split is built, rather than from the full dataset before
any split exists — this avoids leaking test-set statistics into
normalization the model trains on. Verified end-to-end: a synthetic
sample with a zero-filled missing-receiver channel stayed exactly zero
after transform, while real channels were correctly z-scored using
train-only statistics.

## Output layout
```
processed_dataset/
├── split.csv              # sample_id -> train/val/test (+ weight_g, occluded)
├── target_scaler.json     # {"mean": ..., "std": ...} fit on train weight_g
└── csi_scaler.json        # {"amp_mean","amp_std","phase_mean","phase_std"} fit on train CSI

checkpoints/
├── csi_model_best.pt
├── video_model_best.pt
└── fusion_concat_model_best.pt

results/
└── comparison.csv         # CSI-only vs video-only vs fusion, overall + occluded/clean
```
