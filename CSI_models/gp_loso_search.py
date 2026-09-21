"""
LOSO (leave-one-session-out) cross-validated GP kernel hyperparameter
search — Option (a): the LOSO-pooled score IS the final reported result.
There is no additional held-out test set carved out on top of it.

Why LOSO instead of the fixed train/val/test split used by
gp_hyperparameter_search.py: in this dataset, Session_ID is effectively a
stand-in for (weight_g, occluded) — every session maps to exactly one
weight and one occlusion status. Holding out a WHOLE session forces
genuine weight-value generalization (the model never sees that
weight/occlusion combination in training at all) instead of letting it
exploit session-specific background/lighting shortcuts, and avoids the
small-fixed-val-split unreliability found in gp_hyperparameter_search.py
(where the val-optimal nu did not match the test-optimal nu).

Trade-off, stated explicitly rather than hidden: because every session is
used as both training data (in the other 7 folds) and held-out data (in
its own fold), the LOSO-pooled score reported here is a CROSS-VALIDATED
ESTIMATE used to compare configs against each other — it is not an
independent held-out test result the way gp_hyperparameter_search.py's
test split was, and should not be reported as one.

Normalization (WeightScaler, CSIScaler) is refit INSIDE each fold, using
only that fold's training sessions — never the held-out session — to
avoid leaking held-out statistics into the values the model is scored
against. Per-fold data (features, both scalers) is computed ONCE and
reused across every hyperparameter config, since normalization doesn't
depend on the kernel choice — refitting it per (fold, config) would be
~60x redundant work for no benefit.

Metrics are computed on POOLED out-of-fold predictions (every trial's
prediction from whichever fold held its session out, concatenated across
all 8 folds), not averaged per-fold metrics — per-fold R2/MAE on a
handful of trials (some sessions have very few) would be far too unstable
to average meaningfully. Same fix used for the CSI ablation study earlier
in this pipeline.

Usage:
    python gp_loso_search.py
    python gp_loso_search.py --configs-limit 20   # faster, smaller grid
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

import config
import train_datasets as td
import train_utils as tu
import gp_csi_regressor as gpr

LENGTH_SCALE_INITS = [0.5, 1.0, 2.0, 5.0, 10.0]
ALPHAS = [1e-10, 1e-5, 1e-3]
MATERN_NUS = [0.5, 1.5, 2.5]


def load_manifest_with_occlusion():
    manifest = pd.read_csv(config.MANIFEST_PATH)
    manifest["occluded"] = manifest["Visual_Occlusion"].fillna(0).astype(float) > 0
    return manifest


def build_kernel(kernel_type, length_scale_init, nu=None):
    if kernel_type == "rbf":
        base = RBF(length_scale=length_scale_init, length_scale_bounds=(1e-2, 1e3))
    elif kernel_type == "matern":
        base = Matern(length_scale=length_scale_init, length_scale_bounds=(1e-2, 1e3), nu=nu)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}")
    return (ConstantKernel(1.0, (1e-2, 1e2)) * base
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2)))


def _extract_features_targets(rows, csi_scaler, weight_scaler):
    X, y_scaled, weight_g = [], [], []
    for _, row in rows.iterrows():
        raw = np.load(row["npz_path"])["csi_amp_phase"]
        normed = csi_scaler.transform(raw)
        X.append(gpr.extract_gp_features(normed))
        y_scaled.append(weight_scaler.transform(float(row["weight_g"])))
        weight_g.append(float(row["weight_g"]))
    return np.stack(X), np.array(y_scaled), np.array(weight_g)


def precompute_fold_data(manifest):
    """
    One entry per session, holding that session out. Features and both
    scalers are computed once here and reused across every hyperparameter
    config tried below.
    """
    sessions = manifest["Session_ID"].unique()
    folds = []
    for held_out_session in sessions:
        train_rows = manifest[manifest["Session_ID"] != held_out_session]
        test_rows = manifest[manifest["Session_ID"] == held_out_session]
        if len(train_rows) == 0 or len(test_rows) == 0:
            print(f"  [skip fold] '{held_out_session}': no train or no test rows.")
            continue

        weight_scaler = td.WeightScaler.fit(train_rows["weight_g"].to_numpy(dtype=np.float32))
        csi_scaler = td.CSIScaler.fit(train_rows["npz_path"].tolist())

        X_train, y_train, _ = _extract_features_targets(train_rows, csi_scaler, weight_scaler)
        X_test, _, weight_g_test = _extract_features_targets(test_rows, csi_scaler, weight_scaler)

        folds.append({
            "held_out_session": held_out_session,
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "weight_g_test": weight_g_test,
            "occluded_test": test_rows["occluded"].to_numpy(),
            "weight_scaler": weight_scaler,
            "n_train": len(train_rows), "n_test": len(test_rows),
        })
    return folds


def run_loso_for_config(folds, kernel_type, length_scale_init, alpha, nu, seed, n_restarts):
    all_pred_g, all_weight_g, all_occluded = [], [], []
    for fold in folds:
        scaler = StandardScaler().fit(fold["X_train"])
        kernel = build_kernel(kernel_type, length_scale_init, nu)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts,
                                       random_state=seed, normalize_y=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(scaler.transform(fold["X_train"]), fold["y_train"])
        pred_scaled = gp.predict(scaler.transform(fold["X_test"]))
        pred_g = fold["weight_scaler"].inverse_transform(pred_scaled)

        all_pred_g.append(pred_g)
        all_weight_g.append(fold["weight_g_test"])
        all_occluded.append(fold["occluded_test"])
    return np.concatenate(all_pred_g), np.concatenate(all_weight_g), np.concatenate(all_occluded)


def _safe_metrics(pred_g, weight_g, mask):
    if mask.sum() == 0:
        return {"mae_g": float("nan"), "rmse_g": float("nan"), "r2": float("nan")}
    return tu.regression_metrics(pred_g[mask], weight_g[mask])


def score_config(folds, kernel_type, length_scale_init, alpha, nu, seed, n_restarts=3):
    pred_g, weight_g, occluded = run_loso_for_config(folds, kernel_type, length_scale_init, alpha,
                                                       nu, seed, n_restarts)
    overall = _safe_metrics(pred_g, weight_g, np.ones_like(weight_g, dtype=bool))
    clean = _safe_metrics(pred_g, weight_g, ~occluded)
    occ = _safe_metrics(pred_g, weight_g, occluded)
    return overall, clean, occ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-limit", type=int, default=None,
                         help="Randomly subsample the grid to this many configs, for a quicker run")
    parser.add_argument("--final-seeds", type=int, default=5,
                         help="Seeds averaged for the final best-config report "
                              "(the grid search itself uses 1 seed per config for speed)")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/gp_loso_search")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_with_occlusion()
    sessions = sorted(manifest["Session_ID"].unique())
    print(f"{len(manifest)} trials across {len(sessions)} sessions: {sessions}")

    print("\nPrecomputing per-fold features (fit once per fold, reused across every config)...")
    folds = precompute_fold_data(manifest)
    print(f"{len(folds)} usable folds "
          f"(n_train/n_test per fold: {[(f['n_train'], f['n_test']) for f in folds]})")

    configs = [{"kernel_type": "rbf", "length_scale_init": ls, "alpha": a, "nu": None}
               for ls, a in product(LENGTH_SCALE_INITS, ALPHAS)]
    configs += [{"kernel_type": "matern", "length_scale_init": ls, "alpha": a, "nu": nu}
                for ls, a, nu in product(LENGTH_SCALE_INITS, ALPHAS, MATERN_NUS)]

    if args.configs_limit and args.configs_limit < len(configs):
        rng = np.random.default_rng(args.random_seed)
        idx = rng.choice(len(configs), size=args.configs_limit, replace=False)
        configs = [configs[i] for i in idx]
        print(f"Subsampled to {len(configs)} configs (--configs-limit).")

    print(f"\n=== LOSO grid search: {len(configs)} configs x {len(folds)} folds "
          f"= {len(configs) * len(folds)} GP fits ===")
    rows = []
    for i, cfg in enumerate(configs):
        overall, clean, occ = score_config(folds, cfg["kernel_type"], cfg["length_scale_init"],
                                            cfg["alpha"], cfg["nu"], seed=0)
        rows.append({
            **cfg,
            "loso_mae_g": overall["mae_g"], "loso_rmse_g": overall["rmse_g"], "loso_r2": overall["r2"],
            "loso_clean_mae_g": clean["mae_g"], "loso_clean_r2": clean["r2"],
            "loso_occluded_mae_g": occ["mae_g"], "loso_occluded_r2": occ["r2"],
        })
        print(f"  [{i + 1}/{len(configs)}] {cfg['kernel_type']:7s} ls={cfg['length_scale_init']:<5} "
              f"alpha={cfg['alpha']:<8} nu={str(cfg['nu']):5s} "
              f"LOSO_MAE={overall['mae_g']:7.1f}g  R2={overall['r2']:.3f}")

    grid_df = pd.DataFrame(rows).sort_values("loso_mae_g")
    grid_df.to_csv(out_dir / "loso_grid_results.csv", index=False)

    print("\n=== Top 10 configs by pooled LOSO MAE ===")
    print(grid_df.head(10)[["kernel_type", "length_scale_init", "alpha", "nu", "loso_mae_g", "loso_r2"]]
          .to_string(index=False))

    best_rows = grid_df.loc[grid_df.groupby("kernel_type")["loso_mae_g"].idxmin()]
    print("\n=== Best config per kernel family (LOSO-selected) ===")
    print(best_rows[["kernel_type", "length_scale_init", "alpha", "nu", "loso_mae_g", "loso_r2",
                      "loso_clean_mae_g", "loso_occluded_mae_g", "loso_occluded_r2"]].to_string(index=False))

    # Multi-seed re-run of the best configs only — a single seed's LOSO run
    # shouldn't be trusted alone any more than anything else in this pipeline.
    print(f"\n=== Multi-seed ({args.final_seeds}) LOSO re-evaluation of best configs ===")
    final_rows = []
    for _, cfg in best_rows.iterrows():
        for seed in range(args.final_seeds):
            overall, clean, occ = score_config(folds, cfg["kernel_type"], cfg["length_scale_init"],
                                                cfg["alpha"], cfg["nu"], seed=seed, n_restarts=5)
            final_rows.append({
                "kernel_type": cfg["kernel_type"], "seed": seed,
                "loso_mae_g": overall["mae_g"], "loso_rmse_g": overall["rmse_g"], "loso_r2": overall["r2"],
                "loso_clean_mae_g": clean["mae_g"], "loso_clean_r2": clean["r2"],
                "loso_occluded_mae_g": occ["mae_g"], "loso_occluded_r2": occ["r2"],
            })
        print(f"  {cfg['kernel_type']}: done")

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(out_dir / "loso_best_config_multiseed.csv", index=False)

    final_summary = final_df.groupby("kernel_type").agg(
        loso_mae_mean=("loso_mae_g", "mean"), loso_mae_std=("loso_mae_g", "std"),
        loso_r2_mean=("loso_r2", "mean"), loso_r2_std=("loso_r2", "std"),
        loso_clean_mae_mean=("loso_clean_mae_g", "mean"),
        loso_occluded_mae_mean=("loso_occluded_mae_g", "mean"),
        loso_occluded_r2_mean=("loso_occluded_r2", "mean"),
    ).reset_index().sort_values("loso_mae_mean")
    final_summary.to_csv(out_dir / "loso_final_summary.csv", index=False)

    pd.set_option("display.width", 180)
    print("\n=== FINAL: LOSO-pooled cross-validated estimate ===")
    print("(this is a cross-validated estimate from cycling all 8 sessions through the held-out "
          "role — NOT an independent held-out test result; every session was used for both "
          "training and evaluation, just never both at once)")
    print(final_summary.round(2).to_string(index=False))
    print(f"\nSaved -> {out_dir / 'loso_grid_results.csv'}")
    print(f"Saved -> {out_dir / 'loso_best_config_multiseed.csv'}")
    print(f"Saved -> {out_dir / 'loso_final_summary.csv'}")

    try:
        import matplotlib.pyplot as plt

        def _plot_curves(ax):
            rbf_curve = grid_df[grid_df.kernel_type == "rbf"].groupby("length_scale_init")["loso_mae_g"].min()
            ax.plot(rbf_curve.index, rbf_curve.values, marker="s", label="rbf")
            for nu in MATERN_NUS:
                sub = grid_df[(grid_df.kernel_type == "matern") & (grid_df.nu == nu)]
                curve = sub.groupby("length_scale_init")["loso_mae_g"].min()
                ax.plot(curve.index, curve.values, marker="o", label=f"matern (nu={nu})")
            ax.set_xscale("log")
            ax.set_xlabel("initial length_scale")

        # Two panels, same fix as gp_hyperparameter_search.py's plot: a
        # single shared y-axis flattens the competitive comparison into an
        # indistinguishable band whenever one config (typically nu=0.5) is
        # far worse than the rest. The cutoff is a percentile over ALL
        # rows, not a groupby-min — groupby silently drops rows where the
        # group key is NaN (RBF's nu column), which previously computed
        # the cutoff from only the Matern groups.
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
        _plot_curves(ax_full)
        ax_full.set_ylabel("Pooled LOSO MAE (g)")
        ax_full.set_title("Full range")
        ax_full.legend()

        _plot_curves(ax_zoom)
        competitive_max = grid_df["loso_mae_g"].quantile(0.75)
        ax_zoom.set_ylim(grid_df["loso_mae_g"].min() * 0.95, competitive_max * 1.1)
        ax_zoom.set_title("Zoomed to competitive configs")

        fig.suptitle("GP kernel search — LOSO cross-validated (all sessions)")
        fig.tight_layout()
        fig.savefig(out_dir / "loso_grid_plot.png", dpi=130)
        plt.close(fig)
        print(f"Saved -> {out_dir / 'loso_grid_plot.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
