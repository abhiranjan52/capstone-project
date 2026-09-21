"""
Evaluation of CSI-only, Video-only, and Fusion models.

In addition to overall MAE/RMSE/R2, this script calculates:
- Individual MAE for every test sample
- True weight vs predicted weight
- True class vs predicted class
- Misclassified samples
- Most misclassified sample
- Class-wise MAE
- Class-wise RMSE
- Class-wise misclassification count/rate

Outputs:
    results/comparison.csv
    results/per_sample_errors.csv
    results/classwise_errors.csv
"""

import numpy as np
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
    csi, cam1, cam2, target = (
        csi.to(device),
        cam1.to(device),
        cam2.to(device),
        target.to(device),
    )
    return model(csi, cam1, cam2), target, weight_g, occluded


def _get_predictions(model, loader, device, forward_fn, scaler):
    """
    Run model on the complete test set and return sample-level predictions.
    """

    model.eval()

    all_preds = []
    all_targets = []
    all_occluded = []
    all_sample_ids = []

    with __import__("torch").no_grad():
        for batch in loader:

            pred_scaled, target_scaled, weight_g, occluded = forward_fn(
                model, batch, device
            )

            preds_g = scaler.inverse_transform(
                pred_scaled.detach().cpu().numpy()
            )

            targets_g = np.asarray(weight_g, dtype=np.float32)
            occluded_np = np.asarray(occluded, dtype=bool)

            # sample_id is the last item in every dataset batch
            sample_ids = batch[-1]

            if isinstance(sample_ids, (list, tuple)):
                sample_ids = list(sample_ids)
            else:
                sample_ids = [sample_ids]

            all_preds.extend(preds_g.tolist())
            all_targets.extend(targets_g.tolist())
            all_occluded.extend(occluded_np.tolist())
            all_sample_ids.extend(sample_ids)

    return (
        np.asarray(all_preds, dtype=np.float32),
        np.asarray(all_targets, dtype=np.float32),
        np.asarray(all_occluded, dtype=bool),
        all_sample_ids,
    )


def _weight_to_class(weight, class_weights):
    """
    Convert predicted continuous weight to the nearest available
    weight class.
    """

    return min(
        class_weights,
        key=lambda x: abs(float(weight) - float(x))
    )


def _error_analysis(
    model_name,
    preds_g,
    targets_g,
    occluded,
    sample_ids,
):
    """
    Create sample-level and class-level error analysis.
    """

    # Get the actual weight classes present in the test set
    class_weights = sorted(np.unique(targets_g).tolist())

    # Individual absolute error
    absolute_errors = np.abs(preds_g - targets_g)

    rows = []

    for i in range(len(targets_g)):

        true_weight = float(targets_g[i])
        predicted_weight = float(preds_g[i])

        true_class = _weight_to_class(
            true_weight,
            class_weights
        )

        predicted_class = _weight_to_class(
            predicted_weight,
            class_weights
        )

        misclassified = true_class != predicted_class

        rows.append(
            {
                "model": model_name,
                "sample_id": sample_ids[i],
                "true_weight_g": true_weight,
                "predicted_weight_g": predicted_weight,
                "absolute_error_g": float(absolute_errors[i]),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "misclassified": misclassified,
                "occluded": bool(occluded[i]),
            }
        )

    sample_df = pd.DataFrame(rows)

    # Sort so the largest errors are first
    sample_df = sample_df.sort_values(
        "absolute_error_g",
        ascending=False
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Class-wise analysis
    # ---------------------------------------------------------

    class_rows = []

    for true_class in class_weights:

        class_df = sample_df[
            sample_df["true_class"] == true_class
        ]

        n_samples = len(class_df)

        if n_samples == 0:
            continue

        mae = class_df["absolute_error_g"].mean()

        rmse = np.sqrt(
            np.mean(
                class_df["absolute_error_g"] ** 2
            )
        )

        misclassified_count = int(
            class_df["misclassified"].sum()
        )

        misclassification_rate = (
            misclassified_count / n_samples * 100
        )

        class_rows.append(
            {
                "model": model_name,
                "true_class": true_class,
                "n_samples": n_samples,
                "mae_g": mae,
                "rmse_g": rmse,
                "misclassified_count": misclassified_count,
                "misclassification_rate_percent":
                    misclassification_rate,
            }
        )

    class_df = pd.DataFrame(class_rows)

    return sample_df, class_df


def _evaluate_model(
    model_name,
    model,
    loader,
    device,
    forward_fn,
    scaler,
):
    """
    Evaluate one model and perform detailed error analysis.
    """

    # Existing overall evaluation
    overall = tu.evaluate_in_grams(
        model,
        loader,
        device,
        forward_fn,
        scaler,
        occluded_filter=None,
    )

    clean = tu.evaluate_in_grams(
        model,
        loader,
        device,
        forward_fn,
        scaler,
        occluded_filter=False,
    )

    occluded_result = tu.evaluate_in_grams(
        model,
        loader,
        device,
        forward_fn,
        scaler,
        occluded_filter=True,
    )

    # Detailed predictions
    (
        preds_g,
        targets_g,
        occluded,
        sample_ids,
    ) = _get_predictions(
        model,
        loader,
        device,
        forward_fn,
        scaler,
    )

    sample_df, class_df = _error_analysis(
        model_name,
        preds_g,
        targets_g,
        occluded,
        sample_ids,
    )

    return {
        "summary": {
            "model": model_name,
            "overall_mae_g": overall["mae_g"],
            "overall_rmse_g": overall["rmse_g"],
            "overall_r2": overall["r2"],
            "overall_n": overall["n"],
            "clean_mae_g": clean["mae_g"],
            "clean_rmse_g": clean["rmse_g"],
            "clean_r2": clean["r2"],
            "clean_n": clean["n"],
            "occluded_mae_g": occluded_result["mae_g"],
            "occluded_rmse_g": occluded_result["rmse_g"],
            "occluded_r2": occluded_result["r2"],
            "occluded_n": occluded_result["n"],
        },
        "sample_df": sample_df,
        "class_df": class_df,
    }


def main():

    device = tu.get_device()

    print(f"Device: {device}")

    split_df = splits.load_split()

    scaler = td.WeightScaler.load()

    csi_scaler = td.CSIScaler.load()

    results = []

    all_sample_results = []

    all_class_results = []

    # =========================================================
    # CSI ONLY
    # =========================================================

    if config_train.CSI_CHECKPOINT.exists():

        print("\nEvaluating CSI-only...")

        test_ds = td.CSIOnlyDataset(
            split_df,
            scaler,
            csi_scaler,
            "test",
        )

        loader = DataLoader(
            test_ds,
            batch_size=config_train.BATCH_SIZE,
            shuffle=False,
        )

        n_rx = test_ds[0][0].shape[0]

        model = CSIRegressor(n_rx).to(device)

        ckpt = tu.load_checkpoint(
            config_train.CSI_CHECKPOINT,
            map_location=device,
        )

        model.load_state_dict(
            ckpt["model_state"]
        )

        analysis = _evaluate_model(
            "CSI-only",
            model,
            loader,
            device,
            _csi_forward_fn,
            scaler,
        )

        results.append(analysis["summary"])

        all_sample_results.append(
            analysis["sample_df"]
        )

        all_class_results.append(
            analysis["class_df"]
        )

    else:

        print(
            f"Skipping CSI-only: "
            f"{config_train.CSI_CHECKPOINT} not found."
        )

    # =========================================================
    # VIDEO ONLY
    # =========================================================

    if config_train.VIDEO_CHECKPOINT.exists():

        print("\nEvaluating Video-only...")

        test_ds = td.VideoOnlyDataset(
            split_df,
            scaler,
            "test",
        )

        loader = DataLoader(
            test_ds,
            batch_size=config_train.BATCH_SIZE,
            shuffle=False,
        )

        model = VideoRegressor().to(device)

        ckpt = tu.load_checkpoint(
            config_train.VIDEO_CHECKPOINT,
            map_location=device,
        )

        model.load_state_dict(
            ckpt["model_state"]
        )

        analysis = _evaluate_model(
            "Video-only",
            model,
            loader,
            device,
            _video_forward_fn,
            scaler,
        )

        results.append(analysis["summary"])

        all_sample_results.append(
            analysis["sample_df"]
        )

        all_class_results.append(
            analysis["class_df"]
        )

    else:

        print(
            f"Skipping Video-only: "
            f"{config_train.VIDEO_CHECKPOINT} not found."
        )

    # =========================================================
    # FUSION
    # =========================================================

    if config_train.FUSION_CONCAT_CHECKPOINT.exists():

        print("\nEvaluating Fusion (CSI + Video)...")

        test_ds = td.MultimodalDataset(
            split_df,
            scaler,
            csi_scaler,
            "test",
        )

        loader = DataLoader(
            test_ds,
            batch_size=config_train.BATCH_SIZE,
            shuffle=False,
        )

        n_rx = test_ds[0][0].shape[0]

        csi_model = CSIRegressor(n_rx)

        video_model = VideoRegressor()

        model = FusionConcatRegressor(
            csi_model.encoder,
            video_model.encoder,
        ).to(device)

        ckpt = tu.load_checkpoint(
            config_train.FUSION_CONCAT_CHECKPOINT,
            map_location=device,
        )

        model.load_state_dict(
            ckpt["model_state"]
        )

        analysis = _evaluate_model(
            "Fusion (concat, CSI+Video)",
            model,
            loader,
            device,
            _fusion_forward_fn,
            scaler,
        )

        results.append(analysis["summary"])

        all_sample_results.append(
            analysis["sample_df"]
        )

        all_class_results.append(
            analysis["class_df"]
        )

    else:

        print(
            f"Skipping fusion: "
            f"{config_train.FUSION_CONCAT_CHECKPOINT} not found."
        )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    if not results:

        print(
            "No checkpoints found — run "
            "train_csi.py / train_video.py / train_fusion.py first."
        )

        return

    config_train.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Overall comparison
    # ---------------------------------------------------------

    df = pd.DataFrame(results)

    out_path = (
        config_train.RESULTS_DIR
        / "comparison.csv"
    )

    df.to_csv(
        out_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Individual sample errors
    # ---------------------------------------------------------

    sample_errors_df = pd.concat(
        all_sample_results,
        ignore_index=True,
    )

    sample_errors_path = (
        config_train.RESULTS_DIR
        / "per_sample_errors.csv"
    )

    sample_errors_df.to_csv(
        sample_errors_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Class-wise errors
    # ---------------------------------------------------------

    class_errors_df = pd.concat(
        all_class_results,
        ignore_index=True,
    )

    class_errors_path = (
        config_train.RESULTS_DIR
        / "classwise_errors.csv"
    )

    class_errors_df.to_csv(
        class_errors_path,
        index=False,
    )

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    pd.set_option(
        "display.width",
        200,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    print("\n==========================================")
    print("OVERALL TEST SET")
    print("==========================================")

    print(
        df[
            [
                "model",
                "overall_mae_g",
                "overall_rmse_g",
                "overall_r2",
                "overall_n",
            ]
        ]
        .round(
            {
                "overall_mae_g": 1,
                "overall_rmse_g": 1,
                "overall_r2": 3,
            }
        )
        .to_string(index=False)
    )

    # =========================================================
    # MOST MISCLASSIFIED / LARGEST ERROR
    # =========================================================

    print("\n==========================================")
    print("LARGEST INDIVIDUAL ERRORS")
    print("==========================================")

    for model_name in sample_errors_df["model"].unique():

        model_samples = sample_errors_df[
            sample_errors_df["model"] == model_name
        ]

        worst = model_samples.iloc[0]

        print(f"\nModel: {model_name}")

        print(
            f"Sample: {worst['sample_id']}"
        )

        print(
            f"True weight: "
            f"{worst['true_weight_g']:.1f}g"
        )

        print(
            f"Predicted weight: "
            f"{worst['predicted_weight_g']:.1f}g"
        )

        print(
            f"Absolute error: "
            f"{worst['absolute_error_g']:.1f}g"
        )

        print(
            f"True class: "
            f"{worst['true_class']}"
        )

        print(
            f"Predicted class: "
            f"{worst['predicted_class']}"
        )

        print(
            f"Misclassified: "
            f"{worst['misclassified']}"
        )

    # =========================================================
    # MISCLASSIFICATION SUMMARY
    # =========================================================

    print("\n==========================================")
    print("MISCLASSIFICATION SUMMARY")
    print("==========================================")

    for model_name in sample_errors_df["model"].unique():

        model_samples = sample_errors_df[
            sample_errors_df["model"] == model_name
        ]

        total = len(model_samples)

        misclassified = int(
            model_samples["misclassified"].sum()
        )

        rate = (
            misclassified / total * 100
            if total > 0
            else 0
        )

        print(
            f"{model_name}: "
            f"{misclassified}/{total} "
            f"({rate:.1f}%) misclassified"
        )

    # =========================================================
    # CLASS-WISE MAE
    # =========================================================

    print("\n==========================================")
    print("CLASS-WISE MAE")
    print("==========================================")

    print(
        class_errors_df[
            [
                "model",
                "true_class",
                "n_samples",
                "mae_g",
                "rmse_g",
                "misclassified_count",
                "misclassification_rate_percent",
            ]
        ]
        .round(
            {
                "mae_g": 1,
                "rmse_g": 1,
                "misclassification_rate_percent": 1,
            }
        )
        .to_string(index=False)
    )

    # =========================================================
    # CLASS WITH HIGHEST MAE
    # =========================================================

    print("\n==========================================")
    print("HIGHEST CLASS-WISE MAE")
    print("==========================================")

    for model_name in class_errors_df["model"].unique():

        model_classes = class_errors_df[
            class_errors_df["model"] == model_name
        ]

        worst_class = model_classes.loc[
            model_classes["mae_g"].idxmax()
        ]

        print(
            f"{model_name}: "
            f"class {worst_class['true_class']} "
            f"-> MAE = "
            f"{worst_class['mae_g']:.1f}g"
        )

    # =========================================================
    # MOST MISCLASSIFIED CLASS
    # =========================================================

    print("\n==========================================")
    print("MOST MISCLASSIFIED CLASS")
    print("==========================================")

    for model_name in class_errors_df["model"].unique():

        model_classes = class_errors_df[
            class_errors_df["model"] == model_name
        ]

        worst_class = model_classes.loc[
            model_classes["misclassified_count"].idxmax()
        ]

        print(
            f"{model_name}: "
            f"class {worst_class['true_class']} "
            f"-> "
            f"{int(worst_class['misclassified_count'])} "
            f"misclassified samples "
            f"({worst_class['misclassification_rate_percent']:.1f}%)"
        )

    # =========================================================
    # TOP 10 WORST SAMPLES
    # =========================================================

    print("\n==========================================")
    print("TOP 10 WORST SAMPLES")
    print("==========================================")

    print(
        sample_errors_df[
            [
                "model",
                "sample_id",
                "true_weight_g",
                "predicted_weight_g",
                "absolute_error_g",
                "true_class",
                "predicted_class",
                "misclassified",
            ]
        ]
        .head(10)
        .round(
            {
                "true_weight_g": 1,
                "predicted_weight_g": 1,
                "absolute_error_g": 1,
            }
        )
        .to_string(index=False)
    )

    # =========================================================
    # EXISTING OCCLUSION RESULTS
    # =========================================================

    print("\n==========================================")
    print("NON-OCCLUDED SUBSET")
    print("==========================================")

    print(
        df[
            [
                "model",
                "clean_mae_g",
                "clean_rmse_g",
                "clean_r2",
                "clean_n",
            ]
        ]
        .round(
            {
                "clean_mae_g": 1,
                "clean_rmse_g": 1,
                "clean_r2": 3,
            }
        )
        .to_string(index=False)
    )

    print("\n==========================================")
    print("OCCLUDED SUBSET")
    print("==========================================")

    print(
        df[
            [
                "model",
                "occluded_mae_g",
                "occluded_rmse_g",
                "occluded_r2",
                "occluded_n",
            ]
        ]
        .round(
            {
                "occluded_mae_g": 1,
                "occluded_rmse_g": 1,
                "occluded_r2": 3,
            }
        )
        .to_string(index=False)
    )

    # =========================================================
    # FILE LOCATIONS
    # =========================================================

    print("\n==========================================")
    print("FILES SAVED")
    print("==========================================")

    print(
        f"Overall comparison -> {out_path}"
    )

    print(
        f"Individual errors  -> {sample_errors_path}"
    )

    print(
        f"Class-wise errors  -> {class_errors_path}"
    )


if __name__ == "__main__":
    main()