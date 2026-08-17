"""
Stage 2: train the video-only weight regressor to convergence.

Usage:
    python train_video.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config_train
import splits
import train_datasets as td
import train_utils as tu
from models import VideoRegressor


def main():
    device = tu.get_device()
    print(f"Device: {device}")

    split_df = splits.load_split()
    scaler = (td.WeightScaler.load() if config_train.TARGET_SCALER_PATH.exists()
              else td.fit_and_save_scaler(split_df))

    train_ds = td.VideoOnlyDataset(split_df, scaler, "train")
    val_ds = td.VideoOnlyDataset(split_df, scaler, "val")
    test_ds = td.VideoOnlyDataset(split_df, scaler, "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config_train.BATCH_SIZE, shuffle=True,
                               num_workers=config_train.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config_train.BATCH_SIZE, shuffle=False,
                             num_workers=config_train.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False,
                              num_workers=config_train.NUM_WORKERS)

    model = VideoRegressor().to(device)

    def forward_fn(model, batch, device):
        cam1, cam2, target, weight_g, occluded, _sample_id = batch
        cam1, cam2, target = cam1.to(device), cam2.to(device), target.to(device)
        pred = model(cam1, cam2)
        return pred, target, weight_g, occluded

    criterion = nn.SmoothL1Loss()
    # Pretrained backbone (if used) gets a smaller LR than the rest of the model.
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if "encoder.frame_backbone.backbone" in name else other_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config_train.LEARNING_RATE * config_train.VIDEO_BACKBONE_LR_MULT},
        {"params": other_params, "lr": config_train.LEARNING_RATE},
    ], weight_decay=config_train.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config_train.LR_SCHEDULER_FACTOR,
        patience=config_train.LR_SCHEDULER_PATIENCE,
    )
    early_stop = tu.EarlyStopping(patience=config_train.EARLY_STOP_PATIENCE, mode="min")

    print("\n== Training video-only regressor ==")
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
            tu.save_checkpoint(config_train.VIDEO_CHECKPOINT, model.state_dict(),
                                {"epoch": epoch, "val_loss": val_loss})

        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_MAE {val_metrics['mae_g']:.1f}g | val_R2 {val_metrics['r2']:.3f}"
              f"{'  * best' if is_best else ''}")

        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch} (no improvement for {config_train.EARLY_STOP_PATIENCE} epochs).")
            break

    ckpt = tu.load_checkpoint(config_train.VIDEO_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = tu.evaluate_in_grams(model, test_loader, device, forward_fn, scaler)
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")
    print(f"Test set (video-only): MAE={test_metrics['mae_g']:.1f}g RMSE={test_metrics['rmse_g']:.1f}g "
          f"R2={test_metrics['r2']:.3f} (n={test_metrics['n']})")
    print(f"Checkpoint saved -> {config_train.VIDEO_CHECKPOINT}")


if __name__ == "__main__":
    main()
