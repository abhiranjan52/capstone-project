"""
Builds and saves a train/val/test split over the manifest produced by
build_dataset.py, stratified by (weight_g, occluded) so that:
  - every split sees a realistic mix of occluded and non-occluded samples
    during TRAINING (the model is never told which is which), and
  - the TEST split has enough of each to compare occluded vs non-occluded
    performance afterwards.

Run directly to (re)generate config_train.SPLIT_PATH:
    python splits.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import config
import config_train


def load_manifest_with_occlusion() -> pd.DataFrame:
    manifest = pd.read_csv(config.MANIFEST_PATH)
    manifest["occluded"] = manifest["Visual_Occlusion"].fillna(0).astype(float) > 0
    return manifest


def _stratify_key(df: pd.DataFrame) -> pd.Series:
    return df["weight_g"].astype(str) + "_" + df["occluded"].astype(str)


def _safe_stratified_split(df, test_size, stratify_col, seed):
    """
    sklearn's train_test_split raises if any stratum has < 2 members. Falls
    back to an unstratified split (with a warning) rather than crashing,
    since a few small classes are expected with this little data.
    """
    counts = df[stratify_col].value_counts()
    if (counts < 2).any():
        print(f"[splits] WARNING: stratum too small for stratification "
              f"({counts[counts < 2].to_dict()}) — falling back to a "
              "non-stratified split for this partition.")
        return train_test_split(df, test_size=test_size, random_state=seed)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[stratify_col])


def build_trial_level_split(manifest: pd.DataFrame) -> pd.DataFrame:
    df = manifest.copy()
    df["strat"] = _stratify_key(df)

    trainval, test = _safe_stratified_split(df, config_train.TEST_FRACTION, "strat", config_train.RANDOM_SEED)
    trainval = trainval.copy()
    trainval["strat"] = _stratify_key(trainval)  # recompute post-split, same values
    train, val = _safe_stratified_split(trainval, config_train.VAL_FRACTION, "strat", config_train.RANDOM_SEED)

    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def build_session_level_split(manifest: pd.DataFrame) -> pd.DataFrame:
    """Group-holdout by Session_ID. Only safe with enough sessions per stratum."""
    sessions = manifest.groupby("Session_ID").agg(
        weight_g=("weight_g", "first"), occluded=("occluded", "first")
    ).reset_index()
    sessions["strat"] = _stratify_key(sessions)

    counts = sessions["strat"].value_counts()
    if (counts < 2).any():
        print("[splits] SPLIT_STRATEGY='session' requested but strata are too "
              "small for a clean session-level holdout "
              f"({counts[counts < 2].to_dict()}). Falling back to trial-level split.")
        return build_trial_level_split(manifest)

    trainval_s, test_s = train_test_split(sessions, test_size=config_train.TEST_FRACTION,
                                           random_state=config_train.RANDOM_SEED, stratify=sessions["strat"])
    train_s, val_s = train_test_split(trainval_s, test_size=config_train.VAL_FRACTION,
                                       random_state=config_train.RANDOM_SEED, stratify=trainval_s["strat"])

    split_map = {}
    split_map.update({s: "train" for s in train_s["Session_ID"]})
    split_map.update({s: "val" for s in val_s["Session_ID"]})
    split_map.update({s: "test" for s in test_s["Session_ID"]})

    df = manifest.copy()
    df["split"] = df["Session_ID"].map(split_map)
    return df


def build_split() -> pd.DataFrame:
    manifest = load_manifest_with_occlusion()
    if config_train.SPLIT_STRATEGY == "session":
        df = build_session_level_split(manifest)
    else:
        df = build_trial_level_split(manifest)

    config_train.SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["sample_id", "split", "weight_g", "occluded"]].to_csv(config_train.SPLIT_PATH, index=False)
    return df


def load_split() -> pd.DataFrame:
    """Load a previously-built split (rebuilds it if missing)."""
    if not config_train.SPLIT_PATH.exists():
        return build_split()
    manifest = load_manifest_with_occlusion()
    split = pd.read_csv(config_train.SPLIT_PATH)[["sample_id", "split"]]
    return manifest.merge(split, on="sample_id", how="inner")


if __name__ == "__main__":
    df = build_split()
    print(f"Split strategy: {config_train.SPLIT_STRATEGY}")
    print(df.groupby(["split", "occluded"])["sample_id"].count())
    print("\nPer-split weight_g distribution:")
    print(df.groupby(["split", "weight_g"])["sample_id"].count().unstack(fill_value=0))
    print(f"\nSaved -> {config_train.SPLIT_PATH}")
