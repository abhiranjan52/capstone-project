"""
Builds one processed_dataset variant per CSI ablation config — only
csi_amp_phase, since run_csi_ablation.py trains CSIRegressor via
CSIOnlyDataset, which never reads video_camera1/2 from the .npz at all.
Video arrays are therefore NOT copied into these ablation datasets: at
~90MB/sample (2 cameras x 300 padded frames x 224x224x3 uint8) vs
~150KB/sample for CSI, copying unused video data into every config would
multiply total ablation storage by roughly 600x for zero benefit.

Requires build_dataset.py to have already been run once (so
processed_dataset/manifest.csv exists, giving the trial list + labels),
and splits.py to have been run (so the train/val/test split is fixed
before any ablation training happens — see run_csi_ablation.py for why
this matters).

Usage:
    python build_ablation_datasets.py
    python build_ablation_datasets.py --configs raw full leave_out_hampel
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
import csi
import ledger
import csi_ablation as ca

ABLATION_ROOT = Path("processed_dataset_ablation")


def _with_timestamps(base_manifest):
    """
    manifest.csv (from build_dataset.py) doesn't carry Timestamp_Start/
    Timestamp_End — build_csi_features needs them to redefine each
    trial's window, so they're recovered here via a join back to the
    ledger on (Session_ID, Trial_ID).
    """
    led = ledger.load_ledger()[["Session_ID", "Trial_ID", "Timestamp_Start", "Timestamp_End"]]
    merged = base_manifest.merge(led, on=["Session_ID", "Trial_ID"], how="left")
    missing = merged["Timestamp_Start"].isna().sum()
    if missing:
        raise ValueError(f"{missing} manifest rows had no matching ledger row — "
                          "ledger.py or config.LEDGER_PATH may not match what built the manifest.")
    return merged


def build_one_config(config_name, steps, base_manifest, csi_df):
    out_dir = ABLATION_ROOT / config_name
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_dropped = 0
    for _, row in base_manifest.iterrows():
        csi_feat = ca.build_csi_features(csi_df, row, steps)
        if csi_feat is None:
            n_dropped += 1
            continue

        out_path = samples_dir / f"{row['sample_id']}.npz"
        np.savez_compressed(out_path, csi_amp_phase=csi_feat, weight_g=row["weight_g"])
        new_row = row.to_dict()
        new_row["npz_path"] = str(out_path)
        rows.append(new_row)

    out_manifest = pd.DataFrame(rows)
    out_manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"[{config_name}] {len(out_manifest)}/{len(base_manifest)} samples "
          f"({n_dropped} dropped: all receivers under MIN_CSI_PACKETS for this config)")
    return out_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None,
                         help="Subset of config.ABLATION_CONFIGS to build (default: all)")
    args = parser.parse_args()

    configs = (
        {k: ca.ABLATION_CONFIGS[k] for k in args.configs}
        if args.configs else ca.ABLATION_CONFIGS
    )

    base_manifest = pd.read_csv(config.MANIFEST_PATH)
    base_manifest = _with_timestamps(base_manifest)
    csi_df = csi.load_csi_files()

    print(f"Building {len(configs)} ablation dataset variant(s) from {len(base_manifest)} "
          f"base samples (CSI-only — video is not copied, since CSIOnlyDataset never reads it)...\n")
    for name, steps in configs.items():
        build_one_config(name, steps, base_manifest, csi_df)

    print(f"\nDone. Variants under {ABLATION_ROOT}/<config_name>/")


if __name__ == "__main__":
    main()
