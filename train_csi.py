"""
Stage 1: train the CSI-only weight regressor to convergence.

Usage:
    python train_csi.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config_train
import splits
import train_datasets as td
import train_utils as tu
from models import CSIRegressor


def _forward_fn(model, batch, device):
    csi, target, weight_g, occluded, _sample_id = batch
    csi, target = csi.to(device), target.to(device)
    pred = model(csi)
    return pred, target, weight_g, occluded


def main():
    device = tu.get_device()
    print(f"Device: {device}")

    split_df = splits.load_split()
    scaler = (td.WeightScaler.load() if config_train.TARGET_SCALER_PATH.exists()
              else td.fit_and_save_scaler(split_df))
    csi_scaler = (td.CSIScaler.load() if config_train.CSI_SCALER_PATH.exists()
                  else td.fit_and_save_csi_scaler(split_df))

    train_ds = td.CSIOnlyDataset(split_df, scaler, csi_scaler, "train")
    val_ds = td.CSIOnlyDataset(split_df, scaler, csi_scaler, "val")
    test_ds = td.CSIOnlyDataset(split_df, scaler, csi_scaler, "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config_train.BATCH_SIZE, shuffle=True,
                               num_workers=config_train.NUM_WORKERS, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config_train.BATCH_SIZE, shuffle=False,
                             num_workers=config_train.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False,
                              num_workers=config_train.NUM_WORKERS)

    n_rx = train_ds[0][0].shape[0]
    model = CSIRegressor(n_rx).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.LEARNING_RATE,
                                   weight_decay=config_train.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config_train.LR_SCHEDULER_FACTOR,
        patience=config_train.LR_SCHEDULER_PATIENCE,
    )
    early_stop = tu.EarlyStopping(patience=config_train.EARLY_STOP_PATIENCE, mode="min")

    print("\n== Training CSI-only regressor ==")
    for epoch in range(1, config_train.MAX_EPOCHS + 1):
        train_loss, _, _ = tu.run_epoch(model, train_loader, criterion, device, optimizer, _forward_fn)
        val_loss, val_pred_scaled, val_target_scaled = tu.run_epoch(
            model, val_loader, criterion, device, optimizer=None, forward_fn=_forward_fn
        )
        val_metrics = tu.regression_metrics(
            scaler.inverse_transform(val_pred_scaled), scaler.inverse_transform(val_target_scaled)
        )
        scheduler.step(val_loss)

        is_best = early_stop.step(val_loss)
        if is_best:
            tu.save_checkpoint(config_train.CSI_CHECKPOINT, model.state_dict(),
                                {"n_rx": n_rx, "epoch": epoch, "val_loss": val_loss})

        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_MAE {val_metrics['mae_g']:.1f}g | val_R2 {val_metrics['r2']:.3f}"
              f"{'  * best' if is_best else ''}")

        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch} (no improvement for {config_train.EARLY_STOP_PATIENCE} epochs).")
            break

    # --- reload best checkpoint and report final test metrics ---
    ckpt = tu.load_checkpoint(config_train.CSI_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = tu.evaluate_in_grams(model, test_loader, device, _forward_fn, scaler)
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")
    print(f"Test set (CSI-only): MAE={test_metrics['mae_g']:.1f}g RMSE={test_metrics['rmse_g']:.1f}g "
          f"R2={test_metrics['r2']:.3f} (n={test_metrics['n']})")
    print(f"Checkpoint saved -> {config_train.CSI_CHECKPOINT}")


if __name__ == "__main__":
    main()
