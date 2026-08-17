"""
PyTorch Dataset over the preprocessed .npz samples produced by
build_dataset.py. Import into a training script as needed.

Example:
    from dataset import MultimodalWeightDataset
    from torch.utils.data import DataLoader

    ds = MultimodalWeightDataset("processed_dataset/manifest.csv")
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    csi, cam1, cam2, mask1, mask2, y = next(iter(dl))

Video frames are stored padded to a fixed config.MAX_VIDEO_FRAMES (frames
were extracted at a fixed rate, so real length varies per clip). mask1/
mask2 are boolean tensors of shape (MAX_VIDEO_FRAMES,) — True for real
frames, False for padding — intended for attention masking / masked mean
pooling / etc. downstream.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


class MultimodalWeightDataset(Dataset):
    def __init__(self, manifest_path=None, class_list=None, transform=None):
        self.manifest = pd.read_csv(manifest_path or config.MANIFEST_PATH)
        self.class_list = class_list or config.WEIGHT_CLASSES
        self.class_to_idx = {w: i for i, w in enumerate(self.class_list)}
        self.transform = transform  # optional callable applied to (cam1, cam2) frames

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        data = np.load(row["npz_path"])

        csi_t = torch.from_numpy(data["csi_amp_phase"]).float()       # (T, 64, 2)
        cam1 = torch.from_numpy(data["video_camera1"]).float() / 255.0  # (MAX_VIDEO_FRAMES, H, W, 3)
        cam2 = torch.from_numpy(data["video_camera2"]).float() / 255.0

        # (N, H, W, 3) -> (N, 3, H, W), the usual conv input layout
        cam1 = cam1.permute(0, 3, 1, 2)
        cam2 = cam2.permute(0, 3, 1, 2)

        n1 = int(data["n_valid_frames_camera1"])
        n2 = int(data["n_valid_frames_camera2"])
        max_frames = cam1.shape[0]
        mask1 = torch.arange(max_frames) < n1
        mask2 = torch.arange(max_frames) < n2

        if self.transform is not None:
            cam1, cam2 = self.transform(cam1, cam2)

        label = self.class_to_idx[int(data["weight_g"])]
        return csi_t, cam1, cam2, mask1, mask2, label


if __name__ == "__main__":
    import os
    if not os.path.exists(config.MANIFEST_PATH):
        print(f"No manifest found at {config.MANIFEST_PATH}. Run build_dataset.py first.")
    else:
        ds = MultimodalWeightDataset()
        print(f"{len(ds)} samples.")
        csi_t, cam1, cam2, mask1, mask2, label = ds[0]
        print("csi:", csi_t.shape, "cam1:", cam1.shape, "cam2:", cam2.shape,
              "valid frames (cam1/cam2):", mask1.sum().item(), mask2.sum().item(),
              "label:", label)
