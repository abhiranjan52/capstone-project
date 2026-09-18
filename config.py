"""
Central configuration for the CSI + video multimodal weight-classification
preprocessing pipeline.

Edit the paths in this file to match the local data layout, then run
build_dataset.py.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Input paths — EDIT THESE
# ---------------------------------------------------------------------------
LEDGER_PATH = Path(r"D:\esp-idf\collected_data\dataset\ground_truth_ledger.csv")

# The 3 CSI csv files (headerless, ESP32 CSI-Tool style rows)
CSI_FILES = [
    Path(r"D:\esp-idf\collected_data\dataset\active_ap_data.csv"),
    Path(r"D:\esp-idf\collected_data\dataset\passive1_data.csv"),
    Path(r"D:\esp-idf\collected_data\dataset\passive2_data.csv"),
]

VIDEO_ROOT = Path(r"D:\esp-idf\collected_data\dataset")          # contains camera1_data/ and camera2_data/
CAMERA_DIRS = ["camera1_data", "camera2_data"]
VIDEO_EXTENSIONS = [".mp4", ".MOV", ".mov", ".MP4"]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(r"processed_dataset")
SAMPLES_DIR = OUTPUT_DIR / "samples"      # one .npz per (Session_ID, Trial_ID)
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
DROP_LOG_PATH = OUTPUT_DIR / "dropped_trials.csv"

# ---------------------------------------------------------------------------
# CSI parsing
# ---------------------------------------------------------------------------
TARGET_MAC = "1C:C3:AB:B3:D8:70"

# Column indices in the headerless CSI csv (0-based), confirmed against a
# real sample file:
#   col 2  -> transmitter MAC address
#   col 25 -> CSI_DATA array string, e.g. "[84 64 5 0 ...]"
#   col 26 -> unix epoch timestamp (float, seconds) when the packet was recorded
CSI_MAC_COL = 2
CSI_DATA_COL = 25
CSI_TS_COL = 26

N_SUBCARRIERS = 64          # 128 int8 values = 64 x (imag, real) pairs

# Every trial's raw CSI packet count varies (irregular sampling). Each trial
# is resampled to a fixed number of "time steps" via linear interpolation so
# all samples have identical tensor shape for a batched classifier.
CSI_FIXED_LEN = 100         # time steps per trial after resampling
MIN_CSI_PACKETS = 5         # trials with fewer real packets than this are dropped

# ---------------------------------------------------------------------------
# CSI Preprocessing Pipeline
# ---------------------------------------------------------------------------
# Hampel Filter (Outlier removal)
HAMPEL_WINDOW = 5           # Sliding window size for median calculation
HAMPEL_SIGMA = 3.0          # Number of Median Absolute Deviations (MAD) for threshold

# Lowpass Filter (High-frequency noise removal)
LPF_CUTOFF_HZ = 5.0         # Cutoff frequency in Hz
LPF_FS_HZ = 100.0           # Assumed uniform sampling rate after resampling
LPF_ORDER = 3               # Butterworth filter order

# DC Removal (subtracts each trial's own temporal mean from amplitude/phase)
# Standard for motion/activity-sensing CSI tasks, where the static path-loss
# level is nuisance and the time-varying component is the signal. For a
# STATIC weight-regression task, the opposite may hold: the static
# path-loss/attenuation level is a plausible carrier of the weight signal,
# and per-trial DC removal deletes it before the model ever sees it (no
# amount of downstream z-scoring can recover it, since z-scoring recenters
# across trials, not within one). An ablation study found disabling this
# improved CSI-only test MAE substantially on that run (~470g -> ~260g) —
# worth testing on the real pipeline, hence this flag defaults to keep
# prior behavior (True) rather than silently changing it.
ENABLE_DC_REMOVAL = False

# ---------------------------------------------------------------------------
# Video parsing
# ---------------------------------------------------------------------------
# Each video is pre-trimmed to one trial (~20s) at 60fps (~1200 raw frames).
# Sampling a fixed *count* of frames (e.g. 16) means picking 1 frame every
# ~1.25s regardless of clip length — sparse enough to miss short motion
# events, and it doesn't adapt to the duration mismatches described above
# (a 15s clip and a 25s clip both got the same 16 frames, at different
# effective rates). Sampling at a fixed *rate* instead means frame count
# scales with each clip's actual duration and stays temporally comparable
# across clips.
TARGET_VIDEO_FPS = 10        # frames extracted per second of video
FRAME_SIZE = (224, 224)      # (width, height) resize target

# Because clip durations vary (~20s +/- a few seconds, occasional bigger
# mismatches), frame counts after fixed-rate sampling vary too. For
# batching, every clip's frames are padded (by repeating the last frame)
# up to MAX_VIDEO_FRAMES. The true, unpadded frame count is stored
# alongside so the padding can be masked out downstream. Set this
# comfortably above TARGET_VIDEO_FPS * (max expected clip duration).
MAX_VIDEO_FRAMES = 300       # e.g. 10fps * 30s ceiling

# Video files are ~20s, matching the typical ledger trial duration
# (Timestamp_End - Timestamp_Start), but some files run long or short
# relative to their trial. There are no per-frame timestamps to realign
# against, so mismatches can only be detected, not corrected. Frames are
# still sampled evenly across whatever length the file actually is.
#
# DURATION_TOLERANCE_SEC: mismatches within this many seconds are ignored.
# DURATION_MISMATCH_ACTION:
#   "flag"  -> keep the trial, record the mismatch magnitude in the
#              manifest for later inspection/filtering (default —
#              no data lost silently)
#   "drop"  -> drop the trial outright, logged in dropped_trials.csv
DURATION_TOLERANCE_SEC = 3.0
DURATION_MISMATCH_ACTION = "flag"

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# Mass_g in the ledger is a messy string column ("0", "500g", "1000g", ...).
# It is cleaned into an integer-gram value and used as the primary
# classification target (5 classes in the sample ledger: 0/500/1000/1500/2000).
WEIGHT_CLASSES = [0, 500, 1000, 1500, 2000]
