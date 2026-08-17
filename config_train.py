"""
Training-specific configuration for the multimodal weight-regression
pipeline. Kept separate from config.py (the preprocessing config) so
nothing here clobbers preprocessing paths/CSI settings.

Import order in every training script: `import config` (for
MANIFEST_PATH etc.) then `import config_train` (this file, for
hyperparameters).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
SPLIT_PATH = Path("processed_dataset/split.csv")   # sample_id -> split ("train"/"val"/"test")
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15   # of the remaining train+val pool
RANDOM_SEED = 42

# Occlusion is derived from the ledger's Visual_Occlusion column:
# occluded = Visual_Occlusion > 0. This is used ONLY to (a) stratify the
# split so occluded/non-occluded samples are proportionally represented in
# every split, and (b) slice the test set for the final occluded-vs-clean
# comparison. It is never given to the model as a feature.
#
# A ledger with only a handful of sessions (e.g. 8, with only 3 having an
# occlusion variant) is too small for a *session-level* holdout stratified
# by (weight, occlusion) without leaving strata empty. The default is
# therefore to split at the *trial* level (some same-session trials may
# land in both train and test — a real leakage risk given how similar
# trials within one session likely are). Switch to "session" once enough
# sessions exist per stratum for a clean group holdout.
SPLIT_STRATEGY = "trial"   # "trial" or "session"

# ---------------------------------------------------------------------------
# Target scaling
# ---------------------------------------------------------------------------
# Regression on raw grams (0-2000) is harder to optimize than a normalized
# target. Standardize weight_g using TRAIN-split statistics only, invert
# for metric reporting. Saved next to model checkpoints so eval scripts
# don't need to recompute it.
TARGET_SCALER_PATH = Path("processed_dataset/target_scaler.json")

# CSI amplitude/phase z-score normalization, ALSO fit on TRAIN-split
# statistics only (build_dataset.py no longer normalizes CSI itself, to
# avoid leaking test-set statistics into the values the model trains on —
# see fit_scalers.py, train_datasets.CSIScaler).
CSI_SCALER_PATH = Path("processed_dataset/csi_scaler.json")

# ---------------------------------------------------------------------------
# Video frame sub-sampling (at *training* time, not preprocessing time)
# ---------------------------------------------------------------------------
# The .npz files already hold every frame extracted at 10fps (padded to
# MAX_VIDEO_FRAMES). Running a CNN over all ~200 real frames every step is
# expensive, so each training step samples a smaller clip:
#   - train: N_FRAMES_TRAIN frames chosen uniformly-at-random from the
#     valid (non-padded) range — cheap temporal augmentation, different
#     frames each epoch.
#   - val/test: N_FRAMES_EVAL frames chosen evenly spaced across the valid
#     range — deterministic, reproducible metrics.
N_FRAMES_TRAIN = 16
N_FRAMES_EVAL = 16

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
DEVICE = "cuda"   # falls back to "cpu" automatically if no GPU is available
BATCH_SIZE = 8
NUM_WORKERS = 2

CSI_FEATURE_DIM = 128
VIDEO_FRAME_FEATURE_DIM = 256   # projected dim after the per-frame backbone
VIDEO_CLIP_FEATURE_DIM = 256    # after temporal pooling, per camera
FUSION_HIDDEN_DIM = 256

# Use an ImageNet-pretrained ResNet18 as the per-frame backbone if
# torchvision is available (recommended — much faster convergence with
# this little data); otherwise falls back to a small CNN trained from
# scratch.
USE_PRETRAINED_VIDEO_BACKBONE = True
FREEZE_VIDEO_BACKBONE_EARLY_LAYERS = True  # only fine-tune the later ResNet blocks

LEARNING_RATE = 1e-3
VIDEO_BACKBONE_LR_MULT = 0.1   # pretrained backbone gets a smaller LR than the rest
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 12
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 5.0

# ---------------------------------------------------------------------------
# Fusion stage
# ---------------------------------------------------------------------------
# Baseline fusion = simple feature concatenation, tried first before
# trying anything fancier (cross-attention, gating, etc).
#
# FREEZE_BACKBONES_IN_FUSION:
#   True  -> stage-1/2 encoders are frozen; only the new fusion head trains
#            on top of concatenated features (fast, least overfitting risk
#            with this little data, cleanest ablation of "does adding CSI
#            help on top of what video already learned").
#   False -> encoders are fine-tuned jointly with the fusion head at a
#            reduced LR (FUSION_BACKBONE_LR_MULT) — more capacity, more
#            overfitting risk.
FREEZE_BACKBONES_IN_FUSION = True
FUSION_BACKBONE_LR_MULT = 0.1

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path("checkpoints")
CSI_CHECKPOINT = CHECKPOINT_DIR / "csi_model_best.pt"
VIDEO_CHECKPOINT = CHECKPOINT_DIR / "video_model_best.pt"
FUSION_CONCAT_CHECKPOINT = CHECKPOINT_DIR / "fusion_concat_model_best.pt"

RESULTS_DIR = Path("results")
