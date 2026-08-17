"""
Model architectures for the multimodal weight-regression pipeline.

  CSIRegressor    = CSIEncoder + regression head            (stage 1)
  VideoRegressor  = VideoClipEncoder (x2 cameras) + head    (stage 2)
  FusionConcatRegressor = frozen/fine-tuned CSIEncoder + VideoClipEncoder
                           features concatenated + new head (stage 3, baseline fusion)

Every *Encoder returns a flat feature vector (no head) so stage 3 can reuse
the exact stage 1/2 backbones. Every *Regressor wraps an encoder with a
small MLP head so stages 1/2 can be trained standalone.
"""

import torch
import torch.nn as nn

try:
    import torchvision
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

import config_train


# ---------------------------------------------------------------------------
# Regression head (shared)
# ---------------------------------------------------------------------------
class RegressionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# CSI encoder
# ---------------------------------------------------------------------------
class CSIEncoder(nn.Module):
    """
    Input:  (B, n_rx, T, n_subcarriers, 2)  [amplitude, phase]
    Treats (n_rx * 2) as input channels over a (T x n_subcarriers) "image"
    — a standard treatment of CSI amplitude/phase heatmaps for CNN models.
    Output: (B, feature_dim)
    """

    def __init__(self, n_rx: int, feature_dim: int = None):
        super().__init__()
        feature_dim = feature_dim or config_train.CSI_FEATURE_DIM
        in_channels = n_rx * 2  # amplitude + phase per receiver

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        # (B, n_rx, T, n_sub, 2) -> (B, n_rx*2, T, n_sub)
        b, n_rx, t, n_sub, _ = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b, n_rx * 2, t, n_sub)
        x = self.conv(x).flatten(1)
        return self.proj(x)


class CSIRegressor(nn.Module):
    def __init__(self, n_rx: int):
        super().__init__()
        self.encoder = CSIEncoder(n_rx)
        self.head = RegressionHead(self.encoder.feature_dim)

    def forward(self, csi):
        return self.head(self.encoder(csi))


# ---------------------------------------------------------------------------
# Video encoder
# ---------------------------------------------------------------------------
class FrameBackbone(nn.Module):
    """Per-frame feature extractor. ResNet18 (ImageNet-pretrained) if
    torchvision is available, else a small CNN trained from scratch."""

    def __init__(self, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or config_train.VIDEO_FRAME_FEATURE_DIM

        use_pretrained = _HAS_TORCHVISION and config_train.USE_PRETRAINED_VIDEO_BACKBONE
        if use_pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                net = torchvision.models.resnet18(weights=weights)
            except Exception as e:
                print(f"[FrameBackbone] WARNING: couldn't download pretrained ResNet18 "
                      f"weights ({e!r}) — falling back to a from-scratch CNN. If this "
                      "persists, check network access / set USE_PRETRAINED_VIDEO_BACKBONE=False.")
                use_pretrained = False

        if use_pretrained:
            raw_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
            self.raw_dim = raw_dim
            self._is_resnet = True
            if config_train.FREEZE_VIDEO_BACKBONE_EARLY_LAYERS:
                for name, p in self.backbone.named_parameters():
                    if not (name.startswith("layer4") or name.startswith("fc")):
                        p.requires_grad = False
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.raw_dim = 128
            self._is_resnet = False

        self.proj = nn.Linear(self.raw_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x: (N, 3, H, W) -> (N, out_dim)
        feat = self.backbone(x)
        return self.proj(feat)


class VideoClipEncoder(nn.Module):
    """
    Runs FrameBackbone over sampled frames from BOTH cameras, masked-mean-
    pools each camera over time, concatenates the two camera features.
    Input: cam1, cam2 each (B, K, 3, H, W); this dataset's loader doesn't
    produce a mask (frames are already validity-filtered at sampling time
    — see train_datasets._sample_frame_indices), so no masking is needed
    here; every sampled frame is real.
    Output: (B, 2 * frame_feature_dim)
    """

    def __init__(self, feature_dim: int = None):
        super().__init__()
        feature_dim = feature_dim or config_train.VIDEO_CLIP_FEATURE_DIM
        self.frame_backbone = FrameBackbone(out_dim=config_train.VIDEO_FRAME_FEATURE_DIM)
        self.temporal_pool_proj = nn.Linear(config_train.VIDEO_FRAME_FEATURE_DIM, feature_dim)
        self.feature_dim = feature_dim * 2  # both cameras concatenated

    def _encode_camera(self, cam):
        b, k, c, h, w = cam.shape
        frames = cam.reshape(b * k, c, h, w)
        feats = self.frame_backbone(frames).reshape(b, k, -1)  # (B, K, frame_feat)
        pooled = feats.mean(dim=1)                              # (B, frame_feat)
        return self.temporal_pool_proj(pooled)                  # (B, feature_dim)

    def forward(self, cam1, cam2):
        f1 = self._encode_camera(cam1)
        f2 = self._encode_camera(cam2)
        return torch.cat([f1, f2], dim=-1)


class VideoRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VideoClipEncoder()
        self.head = RegressionHead(self.encoder.feature_dim)

    def forward(self, cam1, cam2):
        return self.head(self.encoder(cam1, cam2))


# ---------------------------------------------------------------------------
# Fusion (baseline: simple concatenation of pretrained encoder features)
# ---------------------------------------------------------------------------
class FusionConcatRegressor(nn.Module):
    """
    Loads the stage-1/2 encoders (weights passed in already loaded from
    checkpoints), concatenates their feature vectors, and trains a new
    regression head on top. Encoders are frozen or fine-tuned depending on
    config_train.FREEZE_BACKBONES_IN_FUSION (set by the caller via
    set_backbone_trainable()).
    """

    def __init__(self, csi_encoder: CSIEncoder, video_encoder: VideoClipEncoder):
        super().__init__()
        self.csi_encoder = csi_encoder
        self.video_encoder = video_encoder
        fused_dim = csi_encoder.feature_dim + video_encoder.feature_dim
        self.head = RegressionHead(fused_dim, hidden_dim=config_train.FUSION_HIDDEN_DIM)

    def set_backbones_trainable(self, trainable: bool):
        for p in self.csi_encoder.parameters():
            p.requires_grad = trainable
        for p in self.video_encoder.parameters():
            p.requires_grad = trainable

    def forward(self, csi, cam1, cam2):
        csi_feat = self.csi_encoder(csi)
        video_feat = self.video_encoder(cam1, cam2)
        fused = torch.cat([csi_feat, video_feat], dim=-1)
        return self.head(fused)


if __name__ == "__main__":
    import splits
    import train_datasets as td

    split_df = splits.load_split()
    scaler = td.WeightScaler.load() if config_train.TARGET_SCALER_PATH.exists() else td.fit_and_save_scaler(split_df)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    csi_ds = td.CSIOnlyDataset(split_df, scaler, "train")
    csi, target, *_ = csi_ds[0]
    n_rx = csi.shape[0]
    csi_model = CSIRegressor(n_rx).to(device)
    out = csi_model(csi.unsqueeze(0).to(device))
    print(f"CSIRegressor: input {csi.shape} -> output {out.shape} (should be (1,))")

    vid_ds = td.VideoOnlyDataset(split_df, scaler, "train")
    cam1, cam2, *_ = vid_ds[0]
    vid_model = VideoRegressor().to(device)
    out = vid_model(cam1.unsqueeze(0).to(device), cam2.unsqueeze(0).to(device))
    print(f"VideoRegressor: input cam1={cam1.shape} cam2={cam2.shape} -> output {out.shape}")

    fusion_model = FusionConcatRegressor(csi_model.encoder, vid_model.encoder).to(device)
    out = fusion_model(csi.unsqueeze(0).to(device), cam1.unsqueeze(0).to(device), cam2.unsqueeze(0).to(device))
    print(f"FusionConcatRegressor: output {out.shape}")
    print(f"Device used: {device}, torchvision available: {_HAS_TORCHVISION}")
