"""
Trains and evaluates the CSI-only CNN baseline (CSIRegressor, from
models.py) against LSTM, RNN, Transformer, VAE (models_csi_variants.py),
and Gaussian Process regression with RBF and Matern kernels
(gp_csi_regressor.py) — all on the SAME train/val/test split (from
splits.py) and the SAME CSIScaler/WeightScaler-normalized features, so
architecture is the only thing that differs between them.

Every torch model (CNN/LSTM/RNN/Transformer/VAE) is trained across
multiple seeds (default 5) — with ~45 training samples, a single seed's
result is not trustworthy on its own; mean +/- std across seeds is
reported. GP fitting is otherwise deterministic given fixed data and
kernel form, so "seed" there only varies the L-BFGS hyperparameter-
optimization restarts (still a legitimate source of run-to-run spread,
just a different one than weight-init/data-order variance).

Usage:
    python evaluate_csi_models.py
    python evaluate_csi_models.py --models cnn lstm gp_rbf --seeds 5
    python evaluate_csi_models.py --max-epochs 100 --patience 15
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config_train
import splits
import train_datasets as td
import train_utils as tu
from models import CSIRegressor
from models_csi_variants import CSILSTMRegressor, CSIRNNRegressor, CSITransformerRegressor, CSIVAERegressor
import gp_csi_regressor as gpr

MODEL_CTORS = {
    "cnn": lambda n_rx: CSIRegressor(n_rx),
    "lstm": lambda n_rx: CSILSTMRegressor(n_rx),
    "rnn": lambda n_rx: CSIRNNRegressor(n_rx),
    "transformer": lambda n_rx: CSITransformerRegressor(n_rx),
    "vae": lambda n_rx: CSIVAERegressor(n_rx),
}
GP_KERNELS = ["gp_rbf", "gp_matern"]


def _forward_fn(model, batch, device):
    csi, target, weight_g, occluded, _sample_id = batch
    csi, target = csi.to(device), target.to(device)
    return model(csi), target, weight_g, occluded


# ---------------------------------------------------------------------------
# Standard training path (CNN / LSTM / RNN / Transformer — single
# regression criterion, reuses train_utils' shared loop)
# ---------------------------------------------------------------------------
def train_and_eval_standard(model_name, split_df, weight_scaler, csi_scaler, seed, device,
                             max_epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "train")
    val_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "val")
    test_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "test")
    train_loader = DataLoader(train_ds, batch_size=config_train.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)

    n_rx = train_ds[0][0].shape[0]
    model = MODEL_CTORS[model_name](n_rx).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.LEARNING_RATE,
                                   weight_decay=config_train.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config_train.LR_SCHEDULER_FACTOR,
        patience=config_train.LR_SCHEDULER_PATIENCE,
    )
    early_stop = tu.EarlyStopping(patience=patience, mode="min")

    best_state, best_val_loss, epoch = None, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        tu.run_epoch(model, train_loader, criterion, device, optimizer, _forward_fn)
        val_loss, _, _ = tu.run_epoch(model, val_loader, criterion, device, optimizer=None, forward_fn=_forward_fn)
        scheduler.step(val_loss)
        if early_stop.step(val_loss):
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_loss = val_loss
        if early_stop.should_stop:
            break

    model.load_state_dict(best_state)
    return _package_result(model_name, seed, epoch, best_val_loss, model, test_loader, device, weight_scaler)


# ---------------------------------------------------------------------------
# VAE training path — compound loss (regression + reconstruction + KL),
# doesn't fit train_utils.run_epoch's single-criterion signature.
# ---------------------------------------------------------------------------
def _vae_epoch(model, loader, device, optimizer, beta_recon, beta_kl):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_batches = 0.0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for csi, target, weight_g, occluded, _sample_id in loader:
            csi, target = csi.to(device), target.to(device)
            mu, logvar = model.encode(csi)
            z = model.reparameterize(mu, logvar) if train_mode else mu
            recon = model.decode(z)
            target_img = model.to_image(csi)
            pred = model.head(mu)

            recon_loss = F.mse_loss(recon, target_img)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            reg_loss = F.smooth_l1_loss(pred, target)
            loss = reg_loss + beta_recon * recon_loss + beta_kl * kl_loss

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config_train.GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_and_eval_vae(split_df, weight_scaler, csi_scaler, seed, device, max_epochs, patience,
                        beta_recon=1.0, beta_kl=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "train")
    val_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "val")
    test_ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, "test")
    train_loader = DataLoader(train_ds, batch_size=config_train.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config_train.BATCH_SIZE, shuffle=False)

    n_rx = train_ds[0][0].shape[0]
    model = CSIVAERegressor(n_rx).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.LEARNING_RATE,
                                   weight_decay=config_train.WEIGHT_DECAY)
    early_stop = tu.EarlyStopping(patience=patience, mode="min")

    best_state, best_val_loss, epoch = None, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        _vae_epoch(model, train_loader, device, optimizer, beta_recon, beta_kl)
        val_loss = _vae_epoch(model, val_loader, device, optimizer=None, beta_recon=beta_recon, beta_kl=beta_kl)
        if early_stop.step(val_loss):
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_loss = val_loss
        if early_stop.should_stop:
            break

    model.load_state_dict(best_state)
    return _package_result("vae", seed, epoch, best_val_loss, model, test_loader, device, weight_scaler)


def _package_result(model_name, seed, epoch, best_val_loss, model, test_loader, device, weight_scaler):
    overall = tu.evaluate_in_grams(model, test_loader, device, _forward_fn, weight_scaler, occluded_filter=None)
    clean = tu.evaluate_in_grams(model, test_loader, device, _forward_fn, weight_scaler, occluded_filter=False)
    occluded = tu.evaluate_in_grams(model, test_loader, device, _forward_fn, weight_scaler, occluded_filter=True)
    return {
        "model": model_name, "seed": seed, "epochs_run": epoch, "best_val_loss": best_val_loss,
        "test_mae_g": overall["mae_g"], "test_rmse_g": overall["rmse_g"], "test_r2": overall["r2"],
        "clean_mae_g": clean["mae_g"], "clean_r2": clean["r2"],
        "occluded_mae_g": occluded["mae_g"], "occluded_r2": occluded["r2"],
        "mean_pred_std_g": np.nan,
    }


# ---------------------------------------------------------------------------
# Gaussian Process path — not a torch model at all; features are
# collapsed over time first (see gp_csi_regressor.py's module docstring
# for why raw flattened features would not work here).
# ---------------------------------------------------------------------------
def _build_gp_arrays(split_df, weight_scaler, csi_scaler, split_name):
    ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, split_name)
    X, y, weight_g, occluded = [], [], [], []
    for i in range(len(ds)):
        csi_t, target, w_g, occ, _sid = ds[i]
        X.append(gpr.extract_gp_features(csi_t.numpy()))
        y.append(float(target))
        weight_g.append(w_g)
        occluded.append(occ)
    return np.stack(X), np.array(y), np.array(weight_g), np.array(occluded, dtype=bool)


def train_and_eval_gp(kernel_name, split_df, weight_scaler, csi_scaler, seed):
    X_train, y_train, _, _ = _build_gp_arrays(split_df, weight_scaler, csi_scaler, "train")
    X_test, y_test_scaled, weight_g_test, occluded_test = _build_gp_arrays(
        split_df, weight_scaler, csi_scaler, "test"
    )

    pred_scaled, std_scaled, _gp = gpr.fit_and_predict(kernel_name.replace("gp_", ""), X_train, y_train,
                                                         X_test, seed)
    pred_g = weight_scaler.inverse_transform(pred_scaled)
    std_g = std_scaled * weight_scaler.std   # std is not a location -> scale only, no mean offset

    def _metrics(mask):
        if mask is None:
            mask = np.ones_like(weight_g_test, dtype=bool)
        if mask.sum() == 0:
            return {"mae_g": float("nan"), "rmse_g": float("nan"), "r2": float("nan")}
        return tu.regression_metrics(pred_g[mask], weight_g_test[mask])

    overall, clean, occluded = _metrics(None), _metrics(~occluded_test), _metrics(occluded_test)
    return {
        "model": kernel_name, "seed": seed, "epochs_run": np.nan, "best_val_loss": np.nan,
        "test_mae_g": overall["mae_g"], "test_rmse_g": overall["rmse_g"], "test_r2": overall["r2"],
        "clean_mae_g": clean["mae_g"], "clean_r2": clean["r2"],
        "occluded_mae_g": occluded["mae_g"], "occluded_r2": occluded["r2"],
        "mean_pred_std_g": float(std_g.mean()),
    }


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                         default=list(MODEL_CTORS.keys()) + GP_KERNELS,
                         help=f"Subset to run (default: all). Choices: {list(MODEL_CTORS.keys()) + GP_KERNELS}")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--out-dir", default="results/csi_model_comparison")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = tu.get_device()
    print(f"Device: {device}")

    split_df = splits.load_split()
    weight_scaler = (td.WeightScaler.load() if config_train.TARGET_SCALER_PATH.exists()
                      else td.fit_and_save_scaler(split_df))
    csi_scaler = (td.CSIScaler.load() if config_train.CSI_SCALER_PATH.exists()
                  else td.fit_and_save_csi_scaler(split_df))

    results = []
    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        for seed in range(args.seeds):
            if model_name in MODEL_CTORS and model_name != "vae":
                r = train_and_eval_standard(model_name, split_df, weight_scaler, csi_scaler, seed,
                                             device, args.max_epochs, args.patience)
            elif model_name == "vae":
                r = train_and_eval_vae(split_df, weight_scaler, csi_scaler, seed, device,
                                        args.max_epochs, args.patience)
            elif model_name in GP_KERNELS:
                r = train_and_eval_gp(model_name, split_df, weight_scaler, csi_scaler, seed)
            else:
                print(f"  [skip] unknown model '{model_name}'")
                continue
            print(f"  seed {seed}: test_MAE={r['test_mae_g']:.1f}g test_R2={r['test_r2']:.3f}")
            results.append(r)

    raw_df = pd.DataFrame(results)
    raw_df.to_csv(out_dir / "per_seed_results.csv", index=False)

    summary = raw_df.groupby("model").agg(
        test_mae_mean=("test_mae_g", "mean"), test_mae_std=("test_mae_g", "std"),
        test_r2_mean=("test_r2", "mean"), test_r2_std=("test_r2", "std"),
        clean_mae_mean=("clean_mae_g", "mean"), occluded_mae_mean=("occluded_mae_g", "mean"),
        mean_pred_std_g=("mean_pred_std_g", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index().sort_values("test_mae_mean")
    summary.to_csv(out_dir / "summary.csv", index=False)

    pd.set_option("display.width", 180)
    print("\n=== CSI-only model comparison (sorted by test MAE, best first) ===")
    print(summary.round(2).to_string(index=False))
    print("\n(mean_pred_std_g is only populated for GP models — the predictive uncertainty "
          "the other architectures don't provide out of the box)")
    print(f"\nSaved -> {out_dir / 'per_seed_results.csv'}")
    print(f"Saved -> {out_dir / 'summary.csv'}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 0.5 * len(summary) + 2))
        y = np.arange(len(summary))
        ax.barh(y, summary["test_mae_mean"], xerr=summary["test_mae_std"].fillna(0), capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(summary["model"])
        ax.invert_yaxis()
        ax.set_xlabel("Test MAE (g), mean +/- std across seeds")
        ax.set_title("CSI-only model comparison — lower is better")
        fig.tight_layout()
        fig.savefig(out_dir / "model_comparison_mae.png", dpi=130)
        plt.close(fig)
        print(f"Saved -> {out_dir / 'model_comparison_mae.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
