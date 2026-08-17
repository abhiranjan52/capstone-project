"""
PyTorch Datasets for training, built on top of the .npz files from
build_dataset.py and the split from splits.py.

Three dataset classes share the same underlying data:
  - CSIOnlyDataset       -> (csi, target, weight_g, occluded, sample_id)
  - VideoOnlyDataset     -> (cam1, cam2, target, weight_g, occluded, sample_id)
  - MultimodalDataset    -> (csi, cam1, cam2, target, weight_g, occluded, sample_id)

`target` is the weight_g label standardized by a WeightScaler fit on the
TRAIN split only (no leakage). `csi` is z-scored by a CSIScaler, also fit
on the TRAIN split only (build_dataset.py no longer normalizes CSI itself
— see fit_scalers.py). `weight_g` (raw grams) and `occluded` are also
returned so evaluation can report metrics in real units and sliced by
occlusion, without ever being visible to the model itself as an input.
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

import config_train


# ---------------------------------------------------------------------------
# Target scaling
# ---------------------------------------------------------------------------
class WeightScaler:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, weights_g: np.ndarray) -> "WeightScaler":
        mean = float(np.mean(weights_g))
        std = float(np.std(weights_g))
        if std < 1e-6:
            std = 1.0
        return cls(mean, std)

    def transform(self, w):
        return (w - self.mean) / self.std

    def inverse_transform(self, w):
        return w * self.std + self.mean

    def save(self, path=None):
        path = path or config_train.TARGET_SCALER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path=None) -> "WeightScaler":
        path = path or config_train.TARGET_SCALER_PATH
        with open(path) as f:
            d = json.load(f)
        return cls(d["mean"], d["std"])


def fit_and_save_scaler(split_df) -> WeightScaler:
    train_weights = split_df.loc[split_df["split"] == "train", "weight_g"].to_numpy(dtype=np.float32)
    scaler = WeightScaler.fit(train_weights)
    scaler.save()
    print(f"[WeightScaler] fit on {len(train_weights)} train samples: "
          f"mean={scaler.mean:.1f}g, std={scaler.std:.1f}g")
    return scaler


# ---------------------------------------------------------------------------
# CSI z-score scaling
# ---------------------------------------------------------------------------
class CSIScaler:
    """
    Z-score normalization for CSI amplitude/phase, fit on the TRAIN split
    only (fixes the earlier leakage where build_dataset.py computed global
    stats across the whole dataset before the split existed).

    Mask-aware: zero-filled entries (missing-receiver channels, see
    csi.py's extract_trial_csi) are excluded from the mean/std computation
    and left at exactly 0 after transform, so they stay a clean "missing"
    marker rather than being shifted to some arbitrary normalized value.
    """

    def __init__(self, amp_mean=0.0, amp_std=1.0, phase_mean=0.0, phase_std=1.0):
        self.amp_mean = amp_mean
        self.amp_std = amp_std
        self.phase_mean = phase_mean
        self.phase_std = phase_std

    @classmethod
    def fit(cls, npz_paths) -> "CSIScaler":
        """
        Streams through each train-split .npz (rather than holding every
        array in memory at once) accumulating sum/sumsq for amplitude and
        phase separately, over valid (non-zero-masked) entries only.
        """
        amp_sum = amp_sumsq = amp_n = 0.0
        phase_sum = phase_sumsq = phase_n = 0.0

        for p in npz_paths:
            arr = np.load(p)["csi_amp_phase"]     # (N_rx, T, 64, 2)
            mask = arr[..., 0] != 0
            amp_vals = arr[..., 0][mask]
            phase_vals = arr[..., 1][mask]

            amp_sum += amp_vals.sum(dtype=np.float64)
            amp_sumsq += np.square(amp_vals, dtype=np.float64).sum()
            amp_n += amp_vals.size

            phase_sum += phase_vals.sum(dtype=np.float64)
            phase_sumsq += np.square(phase_vals, dtype=np.float64).sum()
            phase_n += phase_vals.size

        if amp_n > 0:
            amp_mean = amp_sum / amp_n
            amp_var = max(amp_sumsq / amp_n - amp_mean ** 2, 0.0)
            amp_std = max(float(np.sqrt(amp_var)), 1e-6)
        else:
            amp_mean, amp_std = 0.0, 1.0

        if phase_n > 0:
            phase_mean = phase_sum / phase_n
            phase_var = max(phase_sumsq / phase_n - phase_mean ** 2, 0.0)
            phase_std = max(float(np.sqrt(phase_var)), 1e-6)
        else:
            phase_mean, phase_std = 0.0, 1.0

        return cls(float(amp_mean), amp_std, float(phase_mean), phase_std)

    def transform(self, csi_amp_phase: np.ndarray) -> np.ndarray:
        """csi_amp_phase: (N_rx, T, n_sub, 2) numpy array. Returns a new array."""
        out = csi_amp_phase.copy()
        mask = csi_amp_phase[..., 0] != 0
        out[..., 0] = np.where(mask, (csi_amp_phase[..., 0] - self.amp_mean) / self.amp_std, 0.0)
        out[..., 1] = np.where(mask, (csi_amp_phase[..., 1] - self.phase_mean) / self.phase_std, 0.0)
        return out

    def save(self, path=None):
        path = path or config_train.CSI_SCALER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "amp_mean": self.amp_mean, "amp_std": self.amp_std,
                "phase_mean": self.phase_mean, "phase_std": self.phase_std,
            }, f)

    @classmethod
    def load(cls, path=None) -> "CSIScaler":
        path = path or config_train.CSI_SCALER_PATH
        with open(path) as f:
            d = json.load(f)
        return cls(d["amp_mean"], d["amp_std"], d["phase_mean"], d["phase_std"])


def fit_and_save_csi_scaler(split_df) -> CSIScaler:
    train_paths = split_df.loc[split_df["split"] == "train", "npz_path"].tolist()
    scaler = CSIScaler.fit(train_paths)
    scaler.save()
    print(f"[CSIScaler] fit on {len(train_paths)} train samples: "
          f"amp(mean={scaler.amp_mean:.4f}, std={scaler.amp_std:.4f}), "
          f"phase(mean={scaler.phase_mean:.4f}, std={scaler.phase_std:.4f})")
    return scaler


# ---------------------------------------------------------------------------
# Frame sampling helper (shared by video-containing datasets)
# ---------------------------------------------------------------------------
def _sample_frame_indices(n_valid: int, n_want: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Pick n_want frame indices from [0, n_valid). If n_valid < n_want,
    indices repeat (sampled with replacement / repeated tiling) so the
    output length is always exactly n_want.
    """
    n_valid = max(1, n_valid)
    if train:
        if n_valid >= n_want:
            idx = rng.choice(n_valid, size=n_want, replace=False)
            idx.sort()
        else:
            idx = rng.choice(n_valid, size=n_want, replace=True)
            idx.sort()
    else:
        idx = np.linspace(0, n_valid - 1, n_want).astype(int)
    return idx


def _load_camera_clip(data, cam_key, n_valid_key, n_want, train, rng, frame_size_override=None):
    frames = data[cam_key]                 # (MAX_VIDEO_FRAMES, H, W, 3) uint8
    n_valid = int(data[n_valid_key])
    idx = _sample_frame_indices(n_valid, n_want, train, rng)
    clip = frames[idx].astype(np.float32) / 255.0   # (n_want, H, W, 3)
    return torch.from_numpy(clip).permute(0, 3, 1, 2)  # (n_want, 3, H, W)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class CSIOnlyDataset(Dataset):
    def __init__(self, split_df, scaler: WeightScaler, csi_scaler: "CSIScaler", split_name="train"):
        self.df = split_df[split_df["split"] == split_name].reset_index(drop=True)
        self.scaler = scaler
        self.csi_scaler = csi_scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["npz_path"])
        csi_np = self.csi_scaler.transform(data["csi_amp_phase"])   # (N_rx, T, 64, 2), z-scored
        csi = torch.from_numpy(csi_np).float()
        target = torch.tensor(self.scaler.transform(float(row["weight_g"])), dtype=torch.float32)
        return csi, target, float(row["weight_g"]), bool(row["occluded"]), row["sample_id"]


class VideoOnlyDataset(Dataset):
    def __init__(self, split_df, scaler: WeightScaler, split_name="train", n_frames=None):
        self.df = split_df[split_df["split"] == split_name].reset_index(drop=True)
        self.scaler = scaler
        self.train = split_name == "train"
        self.n_frames = n_frames or (config_train.N_FRAMES_TRAIN if self.train else config_train.N_FRAMES_EVAL)
        self.rng = np.random.default_rng(config_train.RANDOM_SEED + (0 if self.train else 1))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["npz_path"])
        cam1 = _load_camera_clip(data, "video_camera1", "n_valid_frames_camera1",
                                  self.n_frames, self.train, self.rng)
        cam2 = _load_camera_clip(data, "video_camera2", "n_valid_frames_camera2",
                                  self.n_frames, self.train, self.rng)
        target = torch.tensor(self.scaler.transform(float(row["weight_g"])), dtype=torch.float32)
        return cam1, cam2, target, float(row["weight_g"]), bool(row["occluded"]), row["sample_id"]


class MultimodalDataset(Dataset):
    def __init__(self, split_df, scaler: WeightScaler, csi_scaler: "CSIScaler", split_name="train", n_frames=None):
        self.df = split_df[split_df["split"] == split_name].reset_index(drop=True)
        self.scaler = scaler
        self.csi_scaler = csi_scaler
        self.train = split_name == "train"
        self.n_frames = n_frames or (config_train.N_FRAMES_TRAIN if self.train else config_train.N_FRAMES_EVAL)
        self.rng = np.random.default_rng(config_train.RANDOM_SEED + (0 if self.train else 1))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["npz_path"])
        csi_np = self.csi_scaler.transform(data["csi_amp_phase"])
        csi = torch.from_numpy(csi_np).float()
        cam1 = _load_camera_clip(data, "video_camera1", "n_valid_frames_camera1",
                                  self.n_frames, self.train, self.rng)
        cam2 = _load_camera_clip(data, "video_camera2", "n_valid_frames_camera2",
                                  self.n_frames, self.train, self.rng)
        target = torch.tensor(self.scaler.transform(float(row["weight_g"])), dtype=torch.float32)
        return csi, cam1, cam2, target, float(row["weight_g"]), bool(row["occluded"]), row["sample_id"]


if __name__ == "__main__":
    import splits
    split_df = splits.load_split()
    scaler = fit_and_save_scaler(split_df)
    csi_scaler = fit_and_save_csi_scaler(split_df)

    csi_ds = CSIOnlyDataset(split_df, scaler, csi_scaler, "train")
    csi_sample = csi_ds[0]
    print("CSIOnlyDataset sample:", [x.shape if hasattr(x, "shape") else x for x in csi_sample[:2]])
    print(f"  CSI z-scored stats: mean={csi_sample[0].mean():.3f}, std={csi_sample[0].std():.3f}")

    vid_ds = VideoOnlyDataset(split_df, scaler, "train")
    vid_sample = vid_ds[0]
    print("VideoOnlyDataset sample cam1/cam2 shapes:", vid_sample[0].shape, vid_sample[1].shape)

    mm_ds = MultimodalDataset(split_df, scaler, csi_scaler, "train")
    mm_sample = mm_ds[0]
    print("MultimodalDataset sample csi/cam1/cam2 shapes:",
          mm_sample[0].shape, mm_sample[1].shape, mm_sample[2].shape)

    print(f"\ntrain={len(CSIOnlyDataset(split_df, scaler, csi_scaler, 'train'))} "
          f"val={len(CSIOnlyDataset(split_df, scaler, csi_scaler, 'val'))} "
          f"test={len(CSIOnlyDataset(split_df, scaler, csi_scaler, 'test'))}")
