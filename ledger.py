"""
Load and clean the ground-truth ledger.

Responsibilities:
  - Build a (Session_ID, Trial_ID) primary key.
  - Clean the messy Mass_g string column ("0", "500g", ...) into an int.
  - Build a canonical `weight_g` classification label.
  - Keep the other ledger columns (Target_Class, Visual_Occlusion,
    Health_State, Object_Count, Inter_Trial_Perturbation) as auxiliary
    metadata — useful for stratified splitting or multi-task heads later.
"""

import re
import pandas as pd

import config


def _clean_mass(value: str) -> int:
    """'500g' -> 500, '0' -> 0, '2000g' -> 2000."""
    if pd.isna(value):
        raise ValueError("Mass_g is NaN")
    match = re.match(r"^\s*(\d+)\s*g?\s*$", str(value))
    if not match:
        raise ValueError(f"Unrecognized Mass_g value: {value!r}")
    return int(match.group(1))


def load_ledger(path=None) -> pd.DataFrame:
    path = path or config.LEDGER_PATH
    df = pd.read_csv(path)

    required = {
        "Session_ID", "Trial_ID", "Timestamp_Start", "Timestamp_End", "Mass_g",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ledger is missing expected columns: {missing}")

    df["weight_g"] = df["Mass_g"].apply(_clean_mass)

    unexpected = sorted(set(df["weight_g"]) - set(config.WEIGHT_CLASSES))
    if unexpected:
        print(f"[ledger] WARNING: weight classes not in config.WEIGHT_CLASSES: {unexpected}")

    # Primary key used to join CSI + video downstream.
    df["sample_id"] = df["Session_ID"].astype(str) + "__trial" + df["Trial_ID"].astype(str)

    # Basic sanity checks.
    dupes = df.duplicated(subset=["Session_ID", "Trial_ID"])
    if dupes.any():
        raise ValueError(
            f"Ledger has {dupes.sum()} duplicate (Session_ID, Trial_ID) rows — "
            "primary key assumption violated."
        )
    bad_window = df["Timestamp_End"] <= df["Timestamp_Start"]
    if bad_window.any():
        raise ValueError(
            f"{bad_window.sum()} rows have Timestamp_End <= Timestamp_Start."
        )

    return df


if __name__ == "__main__":
    df = load_ledger()
    print(df[["Session_ID", "Trial_ID", "Mass_g", "weight_g", "sample_id"]].head(10))
    print(f"\n{len(df)} trials loaded.")
    print(df["weight_g"].value_counts().sort_index())
