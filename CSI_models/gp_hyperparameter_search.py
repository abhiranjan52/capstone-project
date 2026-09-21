"""
Hyperparameter search over GP kernel configurations for RBF and Matern.

Search axes:
  - length_scale_init: the STARTING value handed to sklearn's L-BFGS
    optimizer. sklearn already optimizes length_scale via marginal-
    likelihood maximization (n_restarts_optimizer draws random
    perturbations around the initial value), but the marginal likelihood
    surface is non-convex — different starting points can converge to
    different local optima. Searching over the starting point is a real,
    separate axis from n_restarts_optimizer, not redundant with it.
  - alpha: GaussianProcessRegressor's fixed diagonal noise/jitter term.
    Distinct from the learned WhiteKernel noise component already in the
    kernel — this is numerical regularization added on top.
  - nu (Matern only): the smoothness parameter. RBF has no equivalent —
    it IS the nu -> infinity limit of Matern, so this axis only applies
    to the Matern family. Sweeps the classic {0.5, 1.5, 2.5} values,
    from least to most smooth.

Model selection uses the VAL split only — never test, to avoid tuning-set
leakage. The single best config per kernel family is then re-evaluated on
TEST across multiple seeds (varying only the L-BFGS restart draws), and
compared against the untuned default config from gp_csi_regressor.py, so
the actual benefit of searching is visible rather than assumed.

Usage:
    python gp_hyperparameter_search.py
    python gp_hyperparameter_search.py --final-seeds 10
"""

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

import splits
import train_datasets as td
import train_utils as tu
import gp_csi_regressor as gpr

LENGTH_SCALE_INITS = [0.5, 1.0, 2.0, 5.0, 10.0]
ALPHAS = [1e-10, 1e-5, 1e-3]
MATERN_NUS = [0.5, 1.5, 2.5]

DEFAULT_CONFIGS = [   # the untuned configs originally shipped in gp_csi_regressor.py
    {"kernel_type": "rbf", "length_scale_init": 1.0, "alpha": 1e-10, "nu": None, "config_label": "default"},
    {"kernel_type": "matern", "length_scale_init": 1.0, "alpha": 1e-10, "nu": 1.5, "config_label": "default"},
]


def build_kernel(kernel_type, length_scale_init, nu=None):
    if kernel_type == "rbf":
        base = RBF(length_scale=length_scale_init, length_scale_bounds=(1e-2, 1e3))
    elif kernel_type == "matern":
        base = Matern(length_scale=length_scale_init, length_scale_bounds=(1e-2, 1e3), nu=nu)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}")
    return (ConstantKernel(1.0, (1e-2, 1e2)) * base
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2)))


def _load_arrays(split_df, weight_scaler, csi_scaler, split_name):
    ds = td.CSIOnlyDataset(split_df, weight_scaler, csi_scaler, split_name)
    X, y_scaled, weight_g, occluded = [], [], [], []
    for i in range(len(ds)):
        csi_t, target, w_g, occ, _sid = ds[i]
        X.append(gpr.extract_gp_features(csi_t.numpy()))
        y_scaled.append(float(target))
        weight_g.append(w_g)
        occluded.append(occ)
    return (np.stack(X), np.array(y_scaled), np.array(weight_g, dtype=np.float64),
            np.array(occluded, dtype=bool))


def fit_one(kernel_type, length_scale_init, alpha, nu, X_train, y_train, seed, n_restarts):
    scaler = StandardScaler().fit(X_train)
    kernel = build_kernel(kernel_type, length_scale_init, nu)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts,
                                   random_state=seed, normalize_y=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(scaler.transform(X_train), y_train)
    return gp, scaler


def predict_grams(gp, scaler, X, weight_scaler, return_std=False):
    if return_std:
        pred_scaled, std_scaled = gp.predict(scaler.transform(X), return_std=True)
        return weight_scaler.inverse_transform(pred_scaled), std_scaled * weight_scaler.std
    pred_scaled = gp.predict(scaler.transform(X))
    return weight_scaler.inverse_transform(pred_scaled), None


def run_grid_search(kernel_type, X_train, y_train, X_val, weight_g_val, weight_scaler):
    nu_grid = MATERN_NUS if kernel_type == "matern" else [None]
    rows = []
    for length_scale_init, alpha, nu in product(LENGTH_SCALE_INITS, ALPHAS, nu_grid):
        try:
            gp, scaler = fit_one(kernel_type, length_scale_init, alpha, nu, X_train, y_train,
                                  seed=0, n_restarts=3)
        except Exception as e:
            print(f"  [skip] ls_init={length_scale_init} alpha={alpha} nu={nu}: {e!r}")
            continue
        pred_g, _ = predict_grams(gp, scaler, X_val, weight_scaler)
        metrics = tu.regression_metrics(pred_g, weight_g_val)
        rows.append({
            "kernel_type": kernel_type, "length_scale_init": length_scale_init,
            "alpha": alpha, "nu": nu, "val_mae_g": metrics["mae_g"], "val_r2": metrics["r2"],
            "log_marginal_likelihood": gp.log_marginal_likelihood_value_,
            "fitted_kernel": str(gp.kernel_),
        })
    return pd.DataFrame(rows)


def final_test_eval(config, X_train, y_train, X_test, weight_g_test, occluded_test, weight_scaler, n_seeds):
    rows = []
    for seed in range(n_seeds):
        gp, scaler = fit_one(config["kernel_type"], config["length_scale_init"], config["alpha"],
                              config["nu"], X_train, y_train, seed=seed, n_restarts=5)
        pred_g, std_g = predict_grams(gp, scaler, X_test, weight_scaler, return_std=True)

        def _metrics(mask):
            if mask.sum() == 0:
                return {"mae_g": float("nan"), "rmse_g": float("nan"), "r2": float("nan")}
            return tu.regression_metrics(pred_g[mask], weight_g_test[mask])

        overall = _metrics(np.ones_like(weight_g_test, dtype=bool))
        clean = _metrics(~occluded_test)
        occluded = _metrics(occluded_test)
        rows.append({
            "kernel_type": config["kernel_type"], "config_label": config["config_label"], "seed": seed,
            "test_mae_g": overall["mae_g"], "test_rmse_g": overall["rmse_g"], "test_r2": overall["r2"],
            "clean_mae_g": clean["mae_g"], "clean_r2": clean["r2"],
            "occluded_mae_g": occluded["mae_g"], "occluded_r2": occluded["r2"],
            "mean_pred_std_g": float(std_g.mean()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-seeds", type=int, default=5)
    parser.add_argument("--out-dir", default="results/gp_hyperparam_search")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_df = splits.load_split()
    weight_scaler = td.WeightScaler.load()
    csi_scaler = td.CSIScaler.load()

    X_train, y_train, _, _ = _load_arrays(split_df, weight_scaler, csi_scaler, "train")
    X_val, _, weight_g_val, _ = _load_arrays(split_df, weight_scaler, csi_scaler, "val")
    X_test, _, weight_g_test, occluded_test = _load_arrays(split_df, weight_scaler, csi_scaler, "test")

    grid_frames = []
    for kernel_type in ["rbf", "matern"]:
        print(f"\n=== Grid search on VAL split: {kernel_type} ===")
        df = run_grid_search(kernel_type, X_train, y_train, X_val, weight_g_val, weight_scaler)
        print(f"  {len(df)} configs evaluated, best val MAE = {df['val_mae_g'].min():.1f}g")
        grid_frames.append(df)

    grid_df = pd.concat(grid_frames, ignore_index=True).sort_values("val_mae_g")
    grid_df.to_csv(out_dir / "grid_search_results.csv", index=False)

    print("\n=== Top 10 configs overall (by val MAE) ===")
    print(grid_df.head(10)[["kernel_type", "length_scale_init", "alpha", "nu", "val_mae_g", "val_r2"]]
          .to_string(index=False))

    best_rows = grid_df.loc[grid_df.groupby("kernel_type")["val_mae_g"].idxmin()].copy()
    best_rows["config_label"] = "best_from_search"
    print("\n=== Best config per kernel family (selected on VAL) ===")
    print(best_rows[["kernel_type", "length_scale_init", "alpha", "nu", "val_mae_g", "val_r2"]]
          .to_string(index=False))

    # Final TEST evaluation: best-found configs AND the original untuned
    # defaults, so the actual benefit of searching is visible directly.
    configs_to_finalize = DEFAULT_CONFIGS + best_rows[
        ["kernel_type", "length_scale_init", "alpha", "nu", "config_label"]
    ].to_dict("records")

    final_frames = []
    for config in configs_to_finalize:
        label = f"{config['kernel_type']}_{config['config_label']}"
        print(f"\n=== Final TEST evaluation: {label} "
              f"(ls_init={config['length_scale_init']}, alpha={config['alpha']}, nu={config['nu']}) ===")
        df = final_test_eval(config, X_train, y_train, X_test, weight_g_test, occluded_test,
                              weight_scaler, args.final_seeds)
        print(f"  test_MAE mean={df['test_mae_g'].mean():.1f}g std={df['test_mae_g'].std():.1f}g")
        final_frames.append(df)

    final_df = pd.concat(final_frames, ignore_index=True)
    final_df.to_csv(out_dir / "final_test_results_per_seed.csv", index=False)

    summary = final_df.groupby(["kernel_type", "config_label"]).agg(
        test_mae_mean=("test_mae_g", "mean"), test_mae_std=("test_mae_g", "std"),
        test_r2_mean=("test_r2", "mean"), test_r2_std=("test_r2", "std"),
        clean_mae_mean=("clean_mae_g", "mean"), occluded_mae_mean=("occluded_mae_g", "mean"),
        mean_pred_std_g=("mean_pred_std_g", "mean"),
    ).reset_index().sort_values("test_mae_mean")
    summary.to_csv(out_dir / "final_summary.csv", index=False)

    pd.set_option("display.width", 180)
    print("\n=== Final summary: searched vs default configs (sorted by test MAE) ===")
    print(summary.round(2).to_string(index=False))
    print(f"\nSaved -> {out_dir / 'grid_search_results.csv'}")
    print(f"Saved -> {out_dir / 'final_test_results_per_seed.csv'}")
    print(f"Saved -> {out_dir / 'final_summary.csv'}")

    try:
        import matplotlib.pyplot as plt

        def _plot_curves(ax):
            rbf_curve = grid_df[grid_df.kernel_type == "rbf"].groupby("length_scale_init")["val_mae_g"].min()
            ax.plot(rbf_curve.index, rbf_curve.values, marker="s", label="rbf")
            for nu in MATERN_NUS:
                sub = grid_df[(grid_df.kernel_type == "matern") & (grid_df.nu == nu)]
                curve = sub.groupby("length_scale_init")["val_mae_g"].min()
                ax.plot(curve.index, curve.values, marker="o", label=f"matern (nu={nu})")
            ax.set_xscale("log")
            ax.set_xlabel("initial length_scale")

        # Two panels: full range (so a poorly-behaved config like a very
        # low nu isn't hidden), and zoomed to the competitive cluster —
        # a single shared y-axis would flatten the competitive comparison
        # into an indistinguishable band if any one config is far worse.
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
        _plot_curves(ax_full)
        ax_full.set_ylabel("Val MAE (g) — best alpha at each point")
        ax_full.set_title("Full range")
        ax_full.legend()

        _plot_curves(ax_zoom)
        # A percentile-based cutoff over ALL rows, not a groupby-min
        # approach: groupby silently drops rows where the group key is
        # NaN (RBF's nu column, since RBF has no nu), which previously
        # meant the "competitive cutoff" was computed only from the
        # Matern groups — including the nu=0.5 outlier itself, defeating
        # the point of zooming past it.
        competitive_max = grid_df["val_mae_g"].quantile(0.75)
        ax_zoom.set_ylim(grid_df["val_mae_g"].min() * 0.95, competitive_max * 1.1)
        ax_zoom.set_title("Zoomed to competitive configs")

        fig.suptitle("GP kernel hyperparameter search (validation split)")
        fig.tight_layout()
        fig.savefig(out_dir / "grid_search_plot.png", dpi=130)
        plt.close(fig)
        print(f"Saved -> {out_dir / 'grid_search_plot.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
