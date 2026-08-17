# CSI + Video Multimodal Weight-Classification Pipeline

Preprocessing pipeline that joins a ground-truth ledger, multi-receiver
CSI capture files, and dual-camera video recordings into a single
per-trial dataset for a multimodal weight classifier.

## Data format assumptions
- **Ledger**: one row per trial, keyed by `(Session_ID, Trial_ID)`.
  `Mass_g` is a messy string column (`"0"`, `"500g"`, ...), cleaned into
  an integer `weight_g` label.
- **CSI files**: headerless, ESP32 CSI-Tool row format. Column 2 = MAC
  address, column 25 = `[84 64 5 0 ...]` (128 int8 values = 64
  subcarriers × [imag, real]), column 26 = unix timestamp. Only rows
  matching the configured target MAC are kept.
- **Video**: two camera folders with identical subfolder structure —
  `<camera_dir>/<Session_ID>/<Trial_ID>.{mp4,MOV}`. Each video file is
  pre-trimmed to a single trial (~20s).
- With 3 CSI receivers logging in parallel, a single receiver's file can
  have large timestamp gaps and not cover every trial by itself — this is
  expected when receivers duty-cycle or log at different sampling
  windows, and is why the pipeline treats all 3 files as separate
  channels rather than merging them into one stream (see `csi.py` below).

## Pipeline stages (files)
1. **`ledger.py`** — loads the ledger, cleans `Mass_g` → `weight_g`,
   builds a `sample_id` primary key from `Session_ID` + `Trial_ID`,
   validates no duplicate keys and no inverted timestamp windows.
2. **`csi.py`** — loads each of the 3 CSI csvs, drops every row whose MAC
   doesn't match the configured target, parses the CSI string into
   complex subcarrier values, and keeps each receiver as a separate
   channel (never pooled into one interleaved stream, since two receivers
   sampling in parallel can see genuinely different CSI at the same
   instant). For each trial, slices each receiver's packets inside
   `[Timestamp_Start, Timestamp_End]` and **resamples (linear
   interpolation) to a fixed length** (`CSI_FIXED_LEN=100` time steps) so
   every trial produces an identically-shaped tensor regardless of how
   many raw packets fell in its window. A receiver with too few packets
   in a trial's window is zero-filled rather than dropping the whole
   trial. Converts complex → `[amplitude, unwrapped phase]`, the standard
   representation for CSI classifiers (raw phase is too noisy to use
   directly), then applies a Hampel filter (outlier removal), a
   Butterworth lowpass filter, and DC removal along the time axis.
3. **`video.py`** — locates `<camera_dir>/<Session_ID>/<Trial_ID>.{mp4,MOV}`
   for both cameras, drops the trial if either is missing, and extracts
   frames at a **fixed rate** (`TARGET_VIDEO_FPS=10`, from source video at
   60fps), resized to 224×224, RGB. Frame count scales with each clip's
   actual duration rather than being a fixed count regardless of length —
   this adapts to duration mismatches between the video and its ledger
   trial, and preserves more motion detail than a small fixed frame count
   would. Frames are padded (by repeating the last real frame) up to
   `MAX_VIDEO_FRAMES=300` so every sample batches to the same tensor
   shape; the true frame count is stored per camera
   (`n_valid_frames_camera1/2`) so the padding can be masked out
   downstream.
   Frames are sampled evenly across the whole file, not re-sliced by
   ledger timestamps, since there is no per-frame timestamp to align
   against.
   **Duration mismatch handling**: some video files run longer or shorter
   than their ledger trial's `(Timestamp_End - Timestamp_Start)`. Since
   realignment isn't possible without per-frame timestamps, the pipeline
   computes each video's actual duration and compares it to the expected
   one. Mismatches beyond `config.DURATION_TOLERANCE_SEC` (default 3s)
   are handled per `config.DURATION_MISMATCH_ACTION`: `"flag"` (default)
   keeps the trial but records the mismatch in the manifest for later
   inspection/filtering; `"drop"` removes it outright into
   `dropped_trials.csv`.
4. **`build_dataset.py`** — orchestrates all of the above: for every
   ledger row, checks video availability, extracts + resamples CSI,
   extracts video frames, and saves one `.npz` per trial plus a
   `manifest.csv` (kept samples + labels) and `dropped_trials.csv`
   (dropped samples + the reason: missing video / insufficient CSI /
   video read error). CSI is saved through the Hampel/lowpass/DC
   filtering pipeline in `csi.py` but **without any normalization** —
   normalization is fit and applied later, in the training pipeline, on
   the train split only (see `README_training.md`).
5. **`dataset.py`** — a simple `torch.utils.data.Dataset` that reads the
   manifest and `.npz` files directly into `(csi_tensor, cam1_tensor,
   cam2_tensor, mask1, mask2, label)` tuples, useful for quick inspection
   of the built dataset. The full training pipeline uses its own dataset
   classes (`train_datasets.py`, see `README_training.md`) with proper
   train/val/test-aware normalization instead.

## How to run
```bash
pip install pandas numpy opencv-python scipy torch

# 1. Edit config.py:
#    - LEDGER_PATH, CSI_FILES -> the 3 csv paths
#    - VIDEO_ROOT -> folder containing camera1/ and camera2/

# 2. Build the dataset
python build_dataset.py

# 3. Sanity-check the loader
python dataset.py
```

## Configurable design decisions (in `config.py`)
- **One sample per ledger trial** (not sub-windowed) — matches the
  ledger's own granularity; can be changed to sliding sub-windows without
  touching the CSI/video extraction logic itself.
- **CSI representation**: amplitude + unwrapped phase per receiver,
  resampled to a fixed 100 time steps × 64 subcarriers. Swap
  `to_amplitude_phase` for raw complex, magnitude-only, or a different
  fixed length as needed.
- **Video representation**: raw resized RGB frames sampled at a fixed
  10fps (not a fixed frame count), padded to `MAX_VIDEO_FRAMES=300` with
  a validity mask, not pretrained-CNN embeddings. This keeps the
  preprocessing pipeline agnostic to model architecture — a 3D-CNN /
  CNN+LSTM can train directly on these frames (using the mask for
  pooling), or a frozen backbone (ResNet/EfficientNet) can be run over
  them as a later, separate step.
- **Trials with fewer than `MIN_CSI_PACKETS=5`** raw packets in a
  receiver's window have that receiver zero-filled; a trial with every
  receiver below this threshold is dropped as unreliable, alongside
  trials missing either camera's video file. Both are logged with a
  reason in `dropped_trials.csv` — nothing is silently discarded.

## Output layout
```
processed_dataset/
├── manifest.csv           # sample_id, weight_g label, metadata, duration
│                           # mismatch info, npz path
├── dropped_trials.csv     # sample_id, reason (missing_video / insufficient_csi_packets
│                           # / duration_mismatch / ...)
└── samples/
    └── <Session_ID>__trial<Trial_ID>.npz
        ├── csi_amp_phase          (n_receivers, 100, 64, 2)  float32, unnormalized
        ├── video_camera1          (300, 224, 224, 3) uint8, padded
        ├── video_camera2          (300, 224, 224, 3) uint8, padded
        ├── n_valid_frames_camera1 int  (real, non-padded frame count)
        ├── n_valid_frames_camera2 int
        └── weight_g               int
```

See `README_training.md` for the downstream model-training pipeline that
consumes this output.
