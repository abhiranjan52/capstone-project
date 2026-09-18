"""
Runs the CSI preprocessing ablation study: trains a fresh CSIRegressor
per (config, seed), all using the IDENTICAL train/val/test split (from
splits.py), so the only thing that varies between runs is which
preprocessing steps were applied to the CSI. Requires
build_ablation_datasets.py to have been run first.

Each config is trained across multiple seeds (default 3) — with ~45
training samples, a single seed's result is noisy enough that a
one-run-per-config table would not be trustworthy; mean +/- std across
seeds is reported instead.

The z-score normalizer (CSIScaler) is re-fit per config, on that config's
own train split, since removing a filtering step shifts the value
distribution it should be fit to. The target scaler (WeightScaler) is
fit once — it only depends on weight_g labels, which are identical
across every config (same trials, same split).

Usage:
    python run_csi_ablation.py
    python run_csi_ablation.py --configs raw full leave_out_hampel --seeds 5
    python run_csi_ablation.py --include-no-norm   # adds a "full, unnormalized" comparison
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config_train
import csi_ablation as ca
import train_datasets as td
import train_utils as tu
from models import CSIRegressor
from build_ablation_datasets import ABLATION_ROOT


def _forward_fn(model, batch, device):
    csi, target, weight_g, occluded, _sample_id = batch
    csi, target = csi.to(device), target.to(device)
    return model(csi), target, weight_g, occluded


def load_ablation_split(config_name):
    manifest = pd.read_csv(ABLATION_ROOT / config_name / "manifest.csv")
    manifest["occluded"] = manifest["Visual_Occlusion"].fillna(0).astype(float) > 0
    split = pd.read_csv(config_train.SPLIT_PATH)[["sample_id", "split"]]
    return manifest.merge(split, on="sample_id", how="inner")


def run_one(config_name, steps, split_df, weight_scaler, seed, device, max_epochs, patience,
            csi_scaler_override=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    csi_scaler = csi_scaler_override or td.CSIScaler.fit(
        split_df.loc[split_df["split"] == "train", "npz_path"].tolist()
    )

    train_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "train")
    val_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "val")
    test_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "test")

    train_loader = DataLoader(train_ds, batch_size=config_train.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)

    n_rx = train_ds[0][0].shape[0]
    model = CSIRegressor(n_rx).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.LEARNING_RATE,
                                   weight_decay=config_train.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config_train.LR_SCHEDULER_FACTOR,
        patience=config_train.LR_SCHEDULER_PATIENCE,
    )
    early_stop = tu.EarlyStopping(patience=patience, mode="min")

    best_state, best_val_loss = None, float("inf")
    for epoch in range(1, max_epochs + 1):
        tu.run_epoch(model, train_loader, criterion, device, optimizer, _forward_fn)
        val_loss, _, _ = tu.run_epoch(model, val_loader, criterion, device, optimizer=None, forward_fn=_forward_fn)
        scheduler.step(val_loss)
        is_best = early_stop.step(val_loss)
        if is_best:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_loss = val_loss
        if early_stop.should_stop:
            break

    model.load_state_dict(best_state)
    overall = tu.evaluate_in_grams(model, test_loader, device, _forward_fn, weight_scaler, occluded_filter=None)
    clean = tu.evaluate_in_grams(model, test_loader, device, _forward_fn, weight_scaler, occluded_filter=False)
    occluded = tu.evaluate_in_grams(model, test_loader, device, _forward_fn, weight_scaler, occluded_filter=True)

    return {
        "config": config_name, "seed": seed, "best_val_loss": best_val_loss, "epochs_run": epoch,
        "test_mae_g": overall["mae_g"], "test_rmse_g": overall["rmse_g"], "test_r2": overall["r2"],
        "clean_mae_g": clean["mae_g"], "clean_r2": clean["r2"],
        "occluded_mae_g": occluded["mae_g"], "occluded_r2": occluded["r2"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None, help="Subset of configs to run (default: all built)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds per config")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--include-no-norm", action="store_true",
                         help="Add a 'full_no_normalization' comparison (full pipeline, identity CSI scaler)")
    parser.add_argument("--out-dir", default="results/csi_ablation")
    args = parser.parse_args()

    from pathlib import Path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = tu.get_device()
    print(f"Device: {device}")

    available = [p.name for p in ABLATION_ROOT.iterdir() if p.is_dir()] if ABLATION_ROOT.exists() else []
    configs_to_run = args.configs or available
    if not configs_to_run:
        raise RuntimeError(f"No ablation datasets found under {ABLATION_ROOT} — run build_ablation_datasets.py first.")

    # WeightScaler depends only on weight_g, identical across all configs/splits — fit once.
    any_split_df = load_ablation_split(configs_to_run[0])
    weight_scaler = td.fit_and_save_scaler(any_split_df)

    results = []
    for config_name in configs_to_run:
        split_df = load_ablation_split(config_name)
        steps = ca.ABLATION_CONFIGS.get(config_name, {})
        print(f"\n=== {config_name} ===")
        for seed in range(args.seeds):
            r = run_one(config_name, steps, split_df, weight_scaler, seed, device,
                        args.max_epochs, args.patience)
            print(f"  seed {seed}: test_MAE={r['test_mae_g']:.1f}g test_R2={r['test_r2']:.3f} "
                  f"(epochs_run={r['epochs_run']})")
            results.append(r)

        if args.include_no_norm and config_name == "full":
            identity_scaler = td.CSIScaler(0.0, 1.0, 0.0, 1.0)
            print(f"\n=== full_no_normalization ===")
            for seed in range(args.seeds):
                r = run_one("full_no_normalization", steps, split_df, weight_scaler, seed, device,
                            args.max_epochs, args.patience, csi_scaler_override=identity_scaler)
                print(f"  seed {seed}: test_MAE={r['test_mae_g']:.1f}g test_R2={r['test_r2']:.3f}")
                results.append(r)

    raw_df = pd.DataFrame(results)
    raw_df.to_csv(out_dir / "per_seed_results.csv", index=False)

    summary = raw_df.groupby("config").agg(
        test_mae_mean=("test_mae_g", "mean"), test_mae_std=("test_mae_g", "std"),
        test_r2_mean=("test_r2", "mean"), test_r2_std=("test_r2", "std"),
        clean_mae_mean=("clean_mae_g", "mean"), occluded_mae_mean=("occluded_mae_g", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index().sort_values("test_mae_mean")
    summary.to_csv(out_dir / "summary.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== Ablation summary (sorted by test MAE, best first) ===")
    print(summary.round(2).to_string(index=False))
    print(f"\nSaved -> {out_dir / 'per_seed_results.csv'}")
    print(f"Saved -> {out_dir / 'summary.csv'}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 0.4 * len(summary) + 2))
        y = np.arange(len(summary))
        ax.barh(y, summary["test_mae_mean"], xerr=summary["test_mae_std"], capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(summary["config"])
        ax.invert_yaxis()
        ax.set_xlabel("Test MAE (g), mean +/- std across seeds")
        ax.set_title("CSI preprocessing ablation — lower is better")
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_mae_comparison.png", dpi=130)
        plt.close(fig)
        print(f"Saved -> {out_dir / 'ablation_mae_comparison.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
