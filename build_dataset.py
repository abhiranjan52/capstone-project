"""
End-to-end preprocessing pipeline: joins the ground-truth ledger with CSI
and video data into one .npz file per trial, ready for a multimodal
weight classifier.

CSI is saved through csi.to_amplitude_phase()'s filtering pipeline
(Hampel outlier removal, lowpass, DC removal) but WITHOUT any global
normalization — normalization is fit on the train split only, later, by
fit_scalers.py, to avoid leaking test-set statistics into it.

Usage:
    1. Edit config.py so the paths point at the real data files/folders.
    2. Run:  python build_dataset.py

Output:
    processed_dataset/
        samples/<sample_id>.npz      # csi_amp_phase (unnormalized), video_camera{1,2},
                                     # n_valid_frames_camera{1,2}, weight_g, ...
        manifest.csv                 # one row per kept sample, with label + metadata
        dropped_trials.csv           # one row per dropped trial, with the reason
"""

import numpy as np
import pandas as pd

import config
import ledger
import csi
import video


def build():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("== Loading ledger ==")
    led = ledger.load_ledger()
    print(f"{len(led)} trials in ledger.\n")

    print("== Loading + filtering CSI files ==")
    csi_df = csi.load_csi_files()
    print(f"{len(csi_df)} total CSI packets from target MAC.\n")

    # --- Main Extraction Loop ---
    manifest_rows = []
    dropped_rows = []

    print("== Building per-trial samples ==")
    for _, row in led.iterrows():
        sample_id = row["sample_id"]
        session_id, trial_id = row["Session_ID"], row["Trial_ID"]

        # --- video availability check (drop if either camera is missing) ---
        if not video.trial_has_all_videos(session_id, trial_id):
            dropped_rows.append({"sample_id": sample_id, "reason": "missing_video"})
            continue

        # --- CSI window extraction ---
        csi_window = csi.extract_trial_csi(csi_df, row["Timestamp_Start"], row["Timestamp_End"])
        if csi_window is None:
            dropped_rows.append({"sample_id": sample_id, "reason": "insufficient_csi_packets"})
            continue
        
        csi_amp_phase = csi.to_amplitude_phase(csi_window)  # (N_rx, T, 64, 2), unnormalized —
        # z-score normalization is applied later, fit on the TRAIN split only
        # (see fit_scalers.py / train_datasets.CSIScaler), not here.

        # --- video frame extraction + duration-mismatch check ---
        expected_duration = row["Timestamp_End"] - row["Timestamp_Start"]
        try:
            frames, n_valid_frames, mismatches = video.load_trial_videos(
                session_id, trial_id, expected_duration_sec=expected_duration
            )
        except Exception as e:
            dropped_rows.append({"sample_id": sample_id, "reason": f"video_read_error: {e}"})
            continue

        worst_mismatch = max(mismatches.values(), key=abs)
        mismatch_flagged = abs(worst_mismatch) > config.DURATION_TOLERANCE_SEC
        if mismatch_flagged and config.DURATION_MISMATCH_ACTION == "drop":
            dropped_rows.append({
                "sample_id": sample_id,
                "reason": f"duration_mismatch: {worst_mismatch:+.1f}s "
                          f"(expected {expected_duration:.1f}s)",
            })
            continue

        # --- persist sample ---
        out_path = config.SAMPLES_DIR / f"{sample_id}.npz"
        np.savez_compressed(
            out_path,
            csi_amp_phase=csi_amp_phase,                            # (N_rx, T_CSI, 64, 2) float32
            video_camera1=frames[config.CAMERA_DIRS[0]],             # (MAX_VIDEO_FRAMES, H, W, 3) uint8
            video_camera2=frames[config.CAMERA_DIRS[1]],             # (MAX_VIDEO_FRAMES, H, W, 3) uint8
            n_valid_frames_camera1=n_valid_frames[config.CAMERA_DIRS[0]],  # int, real (non-padded) count
            n_valid_frames_camera2=n_valid_frames[config.CAMERA_DIRS[1]],
            weight_g=row["weight_g"],
        )

        manifest_rows.append({
            "sample_id": sample_id,
            "Session_ID": session_id,
            "Trial_ID": trial_id,
            "weight_g": row["weight_g"],
            "Target_Class": row.get("Target_Class"),
            "Visual_Occlusion": row.get("Visual_Occlusion"),
            "Health_State": row.get("Health_State"),
            "Object_Count": row.get("Object_Count"),
            "expected_duration_sec": round(expected_duration, 2),
            "video_duration_mismatch_sec": round(worst_mismatch, 2),
            "duration_mismatch_flagged": mismatch_flagged,
            "n_valid_frames_camera1": n_valid_frames[config.CAMERA_DIRS[0]],
            "n_valid_frames_camera2": n_valid_frames[config.CAMERA_DIRS[1]],
            "npz_path": str(out_path),
        })

    manifest = pd.DataFrame(manifest_rows)
    dropped = pd.DataFrame(dropped_rows)
    
    if len(manifest) > 0:
        manifest.to_csv(config.MANIFEST_PATH, index=False)
    if len(dropped) > 0:
        dropped.to_csv(config.DROP_LOG_PATH, index=False)

    print(f"\nKept {len(manifest)}/{len(led)} trials.")
    if len(dropped):
        print(f"Dropped {len(dropped)}:")
        print(dropped["reason"].apply(lambda r: r.split(":")[0]).value_counts())
    if len(manifest) and "duration_mismatch_flagged" in manifest.columns:
        n_flagged = manifest["duration_mismatch_flagged"].sum()
        print(f"\n{n_flagged}/{len(manifest)} kept trials have a video/ledger duration "
              f"mismatch > {config.DURATION_TOLERANCE_SEC}s (see manifest.csv "
              "'video_duration_mismatch_sec' column — kept since "
              "DURATION_MISMATCH_ACTION='flag').")
    print(f"\nManifest -> {config.MANIFEST_PATH}")
    print(f"Dropped log -> {config.DROP_LOG_PATH}")
    print(f"Samples -> {config.SAMPLES_DIR}")

    if len(manifest):
        print("\nClass balance (weight_g):")
        print(manifest["weight_g"].value_counts().sort_index())


if __name__ == "__main__":
    build()