"""
Configurable CSI feature builder for preprocessing ablation studies.

Mirrors csi.py's real pipeline (extract_trial_csi + to_amplitude_phase)
step by step, but with each step individually toggleable via a `steps`
dict. Calls csi.py's actual filter functions (_sanitize_phase,
_hampel_filter, _lowpass_filter, _remove_dc) directly rather than
reimplementing their algorithms, so an ablation run reflects exactly what
the real pipeline's filters do — only the *order/gating* logic is
duplicated here (the same approach used in visualize_csi_preprocessing.py).

Pipeline order (each step gated by steps[<name>], default True):
    phase_sanitize -> [resample, always on] -> time_unwrap -> hampel
    -> lowpass -> dc_removal

z-score normalization is NOT included here — it's a train-split-fit,
dataset-level operation (train_datasets.CSIScaler), not a per-trial
transform, so it's ablated separately at the training stage (see
run_csi_ablation.py: NO_NORMALIZATION config).
"""

import numpy as np

import config
import csi


ALL_STEPS = ["phase_sanitize", "time_unwrap", "hampel", "lowpass", "dc_removal"]

FULL_STEPS = {s: True for s in ALL_STEPS}
RAW_STEPS = {s: False for s in ALL_STEPS}

# --- Leave-one-out family: full pipeline minus exactly one step ---
LEAVE_ONE_OUT_CONFIGS = {
    f"leave_out_{s}": {**FULL_STEPS, s: False} for s in ALL_STEPS
}

# --- Cumulative family: raw -> add one step at a time, in pipeline order ---
CUMULATIVE_CONFIGS = {}
enabled_so_far = {}
for s in ALL_STEPS:
    enabled_so_far = {**enabled_so_far, s: True}
    CUMULATIVE_CONFIGS[f"cumulative_+{s}"] = {**RAW_STEPS, **enabled_so_far}

print(CUMULATIVE_CONFIGS)

# --- Combined set actually run by run_csi_ablation.py ---
ABLATION_CONFIGS = {
    "raw": dict(RAW_STEPS),
    **CUMULATIVE_CONFIGS,   # last one here is numerically identical to "full"
    **LEAVE_ONE_OUT_CONFIGS,
    "full": dict(FULL_STEPS),
}


def _build_one_receiver(window_df, steps):
    """
    Returns (amp, phase) each shape (CSI_FIXED_LEN, N_SUBCARRIERS), or
    None if this receiver has too few packets in the window (caller
    zero-fills, matching csi.py's extract_trial_csi behavior).
    """
    if len(window_df) < config.MIN_CSI_PACKETS:
        return None

    ts = window_df["timestamp"].to_numpy()
    raw_complex = np.stack(window_df["csi"].to_numpy())

    complex_for_interp = csi._sanitize_phase(raw_complex) if steps["phase_sanitize"] else raw_complex

    t_norm = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-9)
    t_query = np.linspace(0, 1, config.CSI_FIXED_LEN)
    real_interp = np.stack([
        np.interp(t_query, t_norm, complex_for_interp[:, k].real)
        for k in range(config.N_SUBCARRIERS)
    ], axis=1)
    imag_interp = np.stack([
        np.interp(t_query, t_norm, complex_for_interp[:, k].imag)
        for k in range(config.N_SUBCARRIERS)
    ], axis=1)
    resampled_complex = (real_interp + 1j * imag_interp).astype(np.complex64)

    amp = np.abs(resampled_complex)
    phase = np.angle(resampled_complex)
    if steps["time_unwrap"]:
        phase = np.unwrap(phase, axis=0)

    amp3d, phase3d = amp[None, ...], phase[None, ...]
    if steps["hampel"]:
        amp3d = csi._hampel_filter(amp3d, config.HAMPEL_WINDOW, config.HAMPEL_SIGMA)
        phase3d = csi._hampel_filter(phase3d, config.HAMPEL_WINDOW, config.HAMPEL_SIGMA)
    if steps["lowpass"]:
        amp3d = csi._lowpass_filter(amp3d, config.LPF_CUTOFF_HZ, config.LPF_FS_HZ, config.LPF_ORDER)
        phase3d = csi._lowpass_filter(phase3d, config.LPF_CUTOFF_HZ, config.LPF_FS_HZ, config.LPF_ORDER)
    if steps["dc_removal"]:
        amp3d = csi._remove_dc(amp3d)
        phase3d = csi._remove_dc(phase3d)

    return amp3d[0], phase3d[0]


def build_csi_features(csi_df, row, steps):
    """
    Builds the (n_rx, CSI_FIXED_LEN, N_SUBCARRIERS, 2) feature array for one
    ledger row, under the given ablation `steps` config. Missing/insufficient
    receivers are zero-filled (matching csi.py's behavior), so masking
    downstream (CSIScaler, etc.) still works unchanged. Returns None only
    if EVERY receiver lacks enough packets (matching extract_trial_csi's
    all-receivers-missing case).
    """
    receiver_ids = sorted(csi_df["receiver_id"].unique())
    amp_stack, phase_stack = [], []
    n_valid_receivers = 0

    for rx_id in receiver_ids:
        window = csi_df[
            (csi_df["timestamp"] >= row["Timestamp_Start"])
            & (csi_df["timestamp"] <= row["Timestamp_End"])
            & (csi_df["receiver_id"] == rx_id)
        ]
        result = _build_one_receiver(window, steps)
        if result is None:
            amp_stack.append(np.zeros((config.CSI_FIXED_LEN, config.N_SUBCARRIERS), dtype=np.float32))
            phase_stack.append(np.zeros((config.CSI_FIXED_LEN, config.N_SUBCARRIERS), dtype=np.float32))
        else:
            n_valid_receivers += 1
            amp_stack.append(result[0])
            phase_stack.append(result[1])

    if n_valid_receivers == 0:
        return None

    amp = np.stack(amp_stack, axis=0)     # (n_rx, T, n_sub)
    phase = np.stack(phase_stack, axis=0)
    return np.stack([amp, phase], axis=-1).astype(np.float32)   # (n_rx, T, n_sub, 2)
