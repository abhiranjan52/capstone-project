"""
Final comparison: evaluates CSI-only, video-only, and fusion(concat)
models (whichever checkpoints exist) on the held-out test set — overall,
and split into occluded vs non-occluded subsets. This is the answer to
"does CSI+video beat video alone, and does either modality struggle more
on occluded samples?"

Requires the relevant train_*.py scripts to have been run first.

Usage:
    python evaluate.py
"""

import json

import pandas as pd
from torch.utils.data import DataLoader

import config_train
import splits
import train_datasets as td
import train_utils as tu
from models import CSIRegressor, VideoRegressor, FusionConcatRegressor


def _csi_forward_fn(model, batch, device):
    csi, target, weight_g, occluded, _sample_id = batch
    csi, target = csi.to(device), target.to(device)
    return model(csi), target, weight_g, occluded


def _video_forward_fn(model, batch, device):
    cam1, cam2, target, weight_g, occluded, _sample_id = batch
    cam1, cam2, target = cam1.to(device), cam2.to(device), target.to(device)
    return model(cam1, cam2), target, weight_g, occluded


def _fusion_forward_fn(model, batch, device):
    csi, cam1, cam2, target, weight_g, occluded, _sample_id = batch
    csi, cam1, cam2, target = csi.to(device), cam1.to(device), cam2.to(device), target.to(device)
    return model(csi, cam1, cam2), target, weight_g, occluded


def _report(name, model, loader, device, forward_fn, scaler):
    overall = tu.evaluate_in_grams(model, loader, device, forward_fn, scaler, occluded_filter=None)
    clean = tu.evaluate_in_grams(model, loader, device, forward_fn, scaler, occluded_filter=False)
    occluded = tu.evaluate_in_grams(model, loader, device, forward_fn, scaler, occluded_filter=True)
    return {
        "model": name,
        "overall_mae_g": overall["mae_g"], "overall_rmse_g": overall["rmse_g"],
        "overall_r2": overall["r2"], "overall_n": overall["n"],
        "clean_mae_g": clean["mae_g"], "clean_rmse_g": clean["rmse_g"],
        "clean_r2": clean["r2"], "clean_n": clean["n"],
        "occluded_mae_g": occluded["mae_g"], "occluded_rmse_g": occluded["rmse_g"],
        "occluded_r2": occluded["r2"], "occluded_n": occluded["n"],
    }


def main():
    device = tu.get_device()
    print(f"Device: {device}")

    split_df = splits.load_split()
    scaler = td.WeightScaler.load()
    csi_scaler = td.CSIScaler.load()

    results = []

    # --- CSI-only ---
    if config_train.CSI_CHECKPOINT.exists():
        test_ds = td.CSIOnlyDataset(split_df, scaler, csi_scaler, "test")
        loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)
        n_rx = test_ds[0][0].shape[0]
        model = CSIRegressor(n_rx).to(device)
        ckpt = tu.load_checkpoint(config_train.CSI_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        results.append(_report("CSI-only", model, loader, device, _csi_forward_fn, scaler))
    else:
        print(f"Skipping CSI-only: {config_train.CSI_CHECKPOINT} not found.")

    # --- Video-only ---
    if config_train.VIDEO_CHECKPOINT.exists():
        test_ds = td.VideoOnlyDataset(split_df, scaler, "test")
        loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)
        model = VideoRegressor().to(device)
        ckpt = tu.load_checkpoint(config_train.VIDEO_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        results.append(_report("Video-only", model, loader, device, _video_forward_fn, scaler))
    else:
        print(f"Skipping video-only: {config_train.VIDEO_CHECKPOINT} not found.")

    # --- Fusion (concat) ---
    if config_train.FUSION_CONCAT_CHECKPOINT.exists():
        test_ds = td.MultimodalDataset(split_df, scaler, csi_scaler, "test")
        loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)
        n_rx = test_ds[0][0].shape[0]
        csi_model = CSIRegressor(n_rx)
        video_model = VideoRegressor()
        model = FusionConcatRegressor(csi_model.encoder, video_model.encoder).to(device)
        ckpt = tu.load_checkpoint(config_train.FUSION_CONCAT_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        results.append(_report("Fusion (concat, CSI+Video)", model, loader, device, _fusion_forward_fn, scaler))
    else:
        print(f"Skipping fusion: {config_train.FUSION_CONCAT_CHECKPOINT} not found.")

    if not results:
        print("No checkpoints found — run train_csi.py / train_video.py / train_fusion.py first.")
        return

    df = pd.DataFrame(results)
    config_train.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config_train.RESULTS_DIR / "comparison.csv"
    df.to_csv(out_path, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Overall test set ===")
    print(df[["model", "overall_mae_g", "overall_rmse_g", "overall_r2", "overall_n"]]
          .round({"overall_mae_g": 1, "overall_rmse_g": 1, "overall_r2": 3}).to_string(index=False))

    print("\n=== Non-occluded subset ===")
    print(df[["model", "clean_mae_g", "clean_rmse_g", "clean_r2", "clean_n"]]
          .round({"clean_mae_g": 1, "clean_rmse_g": 1, "clean_r2": 3}).to_string(index=False))

    print("\n=== Occluded subset ===")
    print(df[["model", "occluded_mae_g", "occluded_rmse_g", "occluded_r2", "occluded_n"]]
          .round({"occluded_mae_g": 1, "occluded_rmse_g": 1, "occluded_r2": 3}).to_string(index=False))

    print(f"\nFull comparison saved -> {out_path}")

    if len(results) >= 2:
        print("\n=== Headline comparisons ===")
        by_name = {r["model"]: r for r in results}
        if "Video-only" in by_name and "Fusion (concat, CSI+Video)" in by_name:
            v = by_name["Video-only"]["overall_mae_g"]
            f = by_name["Fusion (concat, CSI+Video)"]["overall_mae_g"]
            delta = v - f
            direction = "IMPROVES on" if delta > 0 else "is WORSE than"
            print(f"Fusion {direction} video-only: {f:.1f}g vs {v:.1f}g MAE "
                  f"({'-' if delta > 0 else '+'}{abs(delta):.1f}g, {abs(delta) / v * 100:.1f}%)")


if __name__ == "__main__":
    main()
