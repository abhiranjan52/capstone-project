"""
Locate and read per-trial video files from the camera1/ and camera2/
folders, and extract frames at a fixed sampling rate per trial.

Assumed layout (identical under both camera dirs):
    <VIDEO_ROOT>/<camera_dir>/<Session_ID>/<Trial_ID>.<mp4|MOV>

Each video file is pre-trimmed to a single trial (~20s), matching the
typical ledger trial duration, so frames are sampled evenly across the
whole file rather than re-sliced by timestamp. Some files run long or
short relative to their trial's ledger duration — see
check_duration_mismatch() / config.DURATION_TOLERANCE_SEC.
"""

from pathlib import Path

import cv2
import numpy as np

import config


def find_video_file(camera_dir: str, session_id: str, trial_id) -> Path | None:
    """Return the path to the trial's video file for one camera, or None if missing."""
    base = config.VIDEO_ROOT / camera_dir / str(session_id)
    if not base.exists():
        return None
    for ext in config.VIDEO_EXTENSIONS:
        candidate = base / f"{trial_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def trial_has_all_videos(session_id: str, trial_id) -> bool:
    """True only if every camera has a video file for this trial."""
    return all(
        find_video_file(cam, session_id, trial_id) is not None
        for cam in config.CAMERA_DIRS
    )


def get_video_duration_sec(video_path: Path) -> float:
    """Duration in seconds, computed from frame count / fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise IOError(f"Video reports invalid fps ({fps}): {video_path}")
    return total / fps


def check_duration_mismatch(video_path: Path, expected_duration_sec: float) -> float:
    """
    Returns the signed mismatch (video_duration - expected_duration) in
    seconds. Positive => video runs long, negative => video runs short.
    """
    actual = get_video_duration_sec(video_path)
    return actual - expected_duration_sec


def extract_frames(video_path: Path, target_fps=None, frame_size=None, max_frames=None):
    """
    Sample frames at a fixed rate (target_fps) across the whole video file,
    resized to frame_size, returned RGB. Frame count therefore scales with
    each clip's actual duration.

    Padded (by repeating the last real frame) up to max_frames so every
    sample has an identical tensor shape for batching.

    Returns:
        frames: uint8 array of shape (max_frames, H, W, 3)
        n_valid: int, the number of real (non-padded) frames — mask
                 frames[n_valid:] out downstream if the padding should
                 not influence the model.
    """
    target_fps = target_fps or config.TARGET_VIDEO_FPS
    frame_size = frame_size or config.FRAME_SIZE
    max_frames = max_frames or config.MAX_VIDEO_FRAMES

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or src_fps <= 0:
        cap.release()
        raise IOError(f"Video has no readable frames/fps: {video_path}")

    duration_sec = total / src_fps
    n_frames = max(1, round(duration_sec * target_fps))
    if n_frames > max_frames:
        print(f"[video] WARNING: {video_path} wants {n_frames} frames at "
              f"{target_fps}fps ({duration_sec:.1f}s) but MAX_VIDEO_FRAMES="
              f"{max_frames} — truncating. Raise MAX_VIDEO_FRAMES if this "
              "happens often.")
        n_frames = max_frames

    indices = np.linspace(0, total - 1, n_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            # Fall back to the previous successfully read frame if a seek fails
            # (common right at the end of some .MOV files).
            frame = frames[-1] if frames else np.zeros((*frame_size[::-1], 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    n_valid = len(frames)
    if n_valid < max_frames:
        pad_frame = frames[-1]
        frames.extend([pad_frame] * (max_frames - n_valid))

    return np.stack(frames).astype(np.uint8), n_valid


def load_trial_videos(session_id: str, trial_id, expected_duration_sec=None) -> tuple:
    """
    Return:
      frames_out:    {camera_dir: (max_frames, H, W, 3) uint8 array}
      n_valid_out:   {camera_dir: int, real (non-padded) frame count}
      mismatch_out:  {camera_dir: mismatch_sec_or_None}
    for every camera for one trial. mismatch_sec is None if
    expected_duration_sec wasn't provided, otherwise (video_duration -
    expected_duration) in seconds, per camera.
    """
    frames_out = {}
    n_valid_out = {}
    mismatch_out = {}
    for cam in config.CAMERA_DIRS:
        path = find_video_file(cam, session_id, trial_id)
        if path is None:
            raise FileNotFoundError(f"Missing video for {cam}/{session_id}/{trial_id}")
        frames_out[cam], n_valid_out[cam] = extract_frames(path)
        mismatch_out[cam] = (
            check_duration_mismatch(path, expected_duration_sec)
            if expected_duration_sec is not None else None
        )
    return frames_out, n_valid_out, mismatch_out


if __name__ == "__main__":
    if not config.VIDEO_ROOT.exists():
        print(f"No video root found at '{config.VIDEO_ROOT}' — nothing to test locally. "
              "Point config.VIDEO_ROOT at the camera1/camera2 folders to test this module.")
    else:
        import ledger
        led = ledger.load_ledger()
        n_ok, n_missing, n_mismatch = 0, 0, 0
        for _, row in led.iterrows():
            if not trial_has_all_videos(row["Session_ID"], row["Trial_ID"]):
                n_missing += 1
                continue
            n_ok += 1
            expected = row["Timestamp_End"] - row["Timestamp_Start"]
            for cam in config.CAMERA_DIRS:
                path = find_video_file(cam, row["Session_ID"], row["Trial_ID"])
                mismatch = check_duration_mismatch(path, expected)
                if abs(mismatch) > config.DURATION_TOLERANCE_SEC:
                    n_mismatch += 1
                    print(f"  [{row['sample_id']}] {cam}: expected {expected:.1f}s, "
                          f"video is {mismatch:+.1f}s off")
        print(f"\n{n_ok} trials have videos in both cameras, {n_missing} missing at least one.")
        print(f"{n_mismatch} camera-files exceed the {config.DURATION_TOLERANCE_SEC}s duration tolerance.")
