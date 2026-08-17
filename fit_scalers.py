"""
Fits both scalers on the TRAIN split ONLY and saves them:
  - WeightScaler (target standardization)
  - CSIScaler    (CSI amplitude/phase z-score normalization)

This replaces the global normalization that used to live in
build_dataset.py (computed across the whole dataset, before any split
existed — a leakage risk). Normalization now happens here, after the
split, using train-split statistics only, and is applied on-the-fly by
the Dataset classes in train_datasets.py.

Run after splits.py and before any train_*.py script:
    python splits.py
    python fit_scalers.py
    python train_csi.py
    ...
(train_csi.py / train_fusion.py / evaluate.py will also fit+save these
automatically if missing, so this step is optional but recommended for
clarity and to fit both scalers in one explicit place.)
"""

import splits
import train_datasets as td


def main():
    split_df = splits.load_split()
    print(f"Split sizes: {split_df['split'].value_counts().to_dict()}")

    weight_scaler = td.fit_and_save_scaler(split_df)
    csi_scaler = td.fit_and_save_csi_scaler(split_df)

    print(f"\nSaved -> {td.config_train.TARGET_SCALER_PATH}")
    print(f"Saved -> {td.config_train.CSI_SCALER_PATH}")
    return weight_scaler, csi_scaler


if __name__ == "__main__":
    main()
