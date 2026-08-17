"""
Stage 3: baseline multimodal fusion via SIMPLE CONCATENATION of the
stage-1 CSI encoder and stage-2 video encoder features, before trying any
more advanced fusion technique. Loads both stage checkpoints, builds
FusionConcatRegressor, and trains the new head (optionally fine-tuning the
backbones — see config_train.FREEZE_BACKBONES_IN_FUSION).

Requires train_csi.py and train_video.py to have been run first.

Usage:
    python train_fusion.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config_train
import splits
import train_datasets as td
import train_utils as tu
from models import CSIRegressor, VideoRegressor, FusionConcatRegressor


def _load_pretrained_encoders(n_rx, device):
    if not config_train.CSI_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{config_train.CSI_CHECKPOINT} not found — run train_csi.py first."
        )
    if not config_train.VIDEO_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{config_train.VIDEO_CHECKPOINT} not found — run train_video.py first."
        )

    csi_model = CSIRegressor(n_rx)
    csi_ckpt = tu.load_checkpoint(config_train.CSI_CHECKPOINT, map_location=device)
    csi_model.load_state_dict(csi_ckpt["model_state"])

    video_model = VideoRegressor()
    video_ckpt = tu.load_checkpoint(config_train.VIDEO_CHECKPOINT, map_location=device)
    video_model.load_state_dict(video_ckpt["model_state"])

    print(f"Loaded CSI encoder from epoch {csi_ckpt['epoch']} (val_loss {csi_ckpt['val_loss']:.4f})")
    print(f"Loaded video encoder from epoch {video_ckpt['epoch']} (val_loss {video_ckpt['val_loss']:.4f})")

    return csi_model.encoder, video_model.encoder


def main():
    device = tu.get_device()
    print(f"Device: {device}")

    split_df = splits.load_split()
    scaler = (td.WeightScaler.load() if config_train.TARGET_SCALER_PATH.exists()
              else td.fit_and_save_scaler(split_df))
    csi_scaler = (td.CSIScaler.load() if config_train.CSI_SCALER_PATH.exists()
                  else td.fit_and_save_csi_scaler(split_df))

    train_ds = td.MultimodalDataset(split_df, scaler, csi_scaler, "train")
    val_ds = td.MultimodalDataset(split_df, scaler, csi_scaler, "val")
    test_ds = td.MultimodalDataset(split_df, scaler, csi_scaler, "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config_train.BATCH_SIZE, shuffle=True,
                               num_workers=config_train.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config_train.BATCH_SIZE, shuffle=False,
                             num_workers=config_train.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False,
                              num_workers=config_train.NUM_WORKERS)

    n_rx = train_ds[0][0].shape[0]
    csi_encoder, video_encoder = _load_pretrained_encoders(n_rx, device)
    model = FusionConcatRegressor(csi_encoder, video_encoder).to(device)
    model.set_backbones_trainable(not config_train.FREEZE_BACKBONES_IN_FUSION)
    print(f"Backbones {'FROZEN' if config_train.FREEZE_BACKBONES_IN_FUSION else 'FINE-TUNED'} during fusion training.")

    def forward_fn(model, batch, device):
        csi, cam1, cam2, target, weight_g, occluded, _sample_id = batch
        csi, cam1, cam2, target = csi.to(device), cam1.to(device), cam2.to(device), target.to(device)
        pred = model(csi, cam1, cam2)
        return pred, target, weight_g, occluded

    criterion = nn.SmoothL1Loss()

    if config_train.FREEZE_BACKBONES_IN_FUSION:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=config_train.LEARNING_RATE,
                                       weight_decay=config_train.WEIGHT_DECAY)
    else:
        head_params = list(model.head.parameters())
        backbone_params = [p for n, p in model.named_parameters()
                            if p.requires_grad and not n.startswith("head.")]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config_train.LEARNING_RATE * config_train.FUSION_BACKBONE_LR_MULT},
            {"params": head_params, "lr": config_train.LEARNING_RATE},
        ], weight_decay=config_train.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config_train.LR_SCHEDULER_FACTOR,
        patience=config_train.LR_SCHEDULER_PATIENCE,
    )
    early_stop = tu.EarlyStopping(patience=config_train.EARLY_STOP_PATIENCE, mode="min")

    print("\n== Training fusion (concat) regressor ==")
    for epoch in range(1, config_train.MAX_EPOCHS + 1):
        train_loss, _, _ = tu.run_epoch(model, train_loader, criterion, device, optimizer, forward_fn)
        val_loss, val_pred_scaled, val_target_scaled = tu.run_epoch(
            model, val_loader, criterion, device, optimizer=None, forward_fn=forward_fn
        )
        val_metrics = tu.regression_metrics(
            scaler.inverse_transform(val_pred_scaled), scaler.inverse_transform(val_target_scaled)
        )
        scheduler.step(val_loss)

        is_best = early_stop.step(val_loss)
        if is_best:
            tu.save_checkpoint(config_train.FUSION_CONCAT_CHECKPOINT, model.state_dict(),
                                {"n_rx": n_rx, "epoch": epoch, "val_loss": val_loss,
                                 "backbones_frozen": config_train.FREEZE_BACKBONES_IN_FUSION})

        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_MAE {val_metrics['mae_g']:.1f}g | val_R2 {val_metrics['r2']:.3f}"
              f"{'  * best' if is_best else ''}")

        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch} (no improvement for {config_train.EARLY_STOP_PATIENCE} epochs).")
            break

    ckpt = tu.load_checkpoint(config_train.FUSION_CONCAT_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = tu.evaluate_in_grams(model, test_loader, device, forward_fn, scaler)
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")
    print(f"Test set (fusion/concat): MAE={test_metrics['mae_g']:.1f}g RMSE={test_metrics['rmse_g']:.1f}g "
          f"R2={test_metrics['r2']:.3f} (n={test_metrics['n']})")
    print(f"Checkpoint saved -> {config_train.FUSION_CONCAT_CHECKPOINT}")


if __name__ == "__main__":
    main()
