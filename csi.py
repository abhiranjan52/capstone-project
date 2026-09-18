"""
Load, filter, and parse the raw CSI csv files, and extract/resample
per-trial CSI windows keyed by ground-truth timestamps.

Raw row format (headerless, ESP32 CSI-Tool style), 0-based column indices:
  2  -> transmitter MAC address (string)
  25 -> CSI_DATA array string, e.g. "[84 64 5 0 ...]" — 128 int8 values,
        i.e. 64 (imag, real) pairs, one pair per subcarrier
  26 -> unix epoch timestamp (float seconds) the packet was recorded

Only rows whose MAC matches config.TARGET_MAC are kept; everything else
is dropped, per spec.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.ndimage as ndimage

import config


def _parse_csi_string(s: str):
    """'[84 64 5 0 ...]' -> complex128 array of shape (n_subcarriers,), or None if malformed."""
    if not isinstance(s, str):
        return None
        
    vals = np.fromstring(s.strip("[]"), dtype=np.int16, sep=" ")
    
    if vals.size != config.N_SUBCARRIERS * 2:
        return None 
        
    imag = vals[0::2].astype(np.float32)
    real = vals[1::2].astype(np.float32)
    return real + 1j * imag


def _sanitize_phase(csi_complex: np.ndarray) -> np.ndarray:
    """
    Sanitizes phase by calculating unwrapped phase across subcarriers 
    and removing the linear slope (CFO/SFO errors).
    """
    amp = np.abs(csi_complex)
    phase = np.angle(csi_complex)
    
    # Unwrap across subcarriers (axis 1)
    phase_unwrapped = np.unwrap(phase, axis=1)
    k = np.arange(phase_unwrapped.shape[1])
    
    sanitized_phase = np.zeros_like(phase_unwrapped)
    for i in range(phase_unwrapped.shape[0]):
        # Linear fit across subcarriers: y = mx + c
        m, c = np.polyfit(k, phase_unwrapped[i], 1)
        sanitized_phase[i] = phase_unwrapped[i] - (m * k + c)
        
    # Reconstruct complex signal with clean phase
    return amp * np.exp(1j * sanitized_phase)


def load_csi_files(paths=None) -> pd.DataFrame:
    """
    Load and concatenate all CSI csv files, keep only packets from
    TARGET_MAC, parse the CSI array column, and sort by timestamp.
    """
    paths = paths or config.CSI_FILES
    frames = []
    for rx_id, p in enumerate(paths):
        raw = pd.read_csv(p, header=None, low_memory=False)
        
        kept = raw[raw[config.CSI_MAC_COL] == config.TARGET_MAC].copy()
        dropped = len(raw) - len(kept)
        
        kept["timestamp"] = kept[config.CSI_TS_COL].astype(float)
        kept["csi"] = kept[config.CSI_DATA_COL].apply(_parse_csi_string)
        
        valid_mask = kept["csi"].notna()
        malformed_count = len(kept) - valid_mask.sum()
        kept = kept[valid_mask]
        
        print(f"[csi] Receiver {rx_id} ({p}): kept {len(kept)}/{len(raw)} rows "
              f"(dropped {dropped} MAC, {malformed_count} malformed)")
              
        kept["receiver_id"] = rx_id
        frames.append(kept[["timestamp", "csi", "receiver_id"]])

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def extract_trial_csi(csi_df: pd.DataFrame, t_start: float, t_end: float) -> np.ndarray:
    """
    Slice CSI packets in [t_start, t_end]. Sanitizes phase, then linearly 
    interpolates timestamps for EACH receiver independently.
    
    Returns zero-masked arrays for receivers without data. Discards trial
    (returns None) if ALL receivers lack sufficient data.
    """
    window = csi_df[(csi_df["timestamp"] >= t_start) & (csi_df["timestamp"] <= t_end)]
    
    receiver_ids = sorted(csi_df["receiver_id"].unique())
    receiver_stacks = []
    valid_receivers = 0

    for rx_id in receiver_ids:
        rx_window = window[window["receiver_id"] == rx_id]
        
        if len(rx_window) < config.MIN_CSI_PACKETS:
            rx_complex = np.zeros(
                (config.CSI_FIXED_LEN, config.N_SUBCARRIERS), 
                dtype=np.complex64
            )
        else:
            valid_receivers += 1
            ts = rx_window["timestamp"].to_numpy()
            csi_stack = np.stack(rx_window["csi"].to_numpy())  # (n_packets, n_subcarriers)
            
            # 1. Sanitize phase on raw subcarriers
            csi_clean = _sanitize_phase(csi_stack)

            # 2. Linear Interpolation using exact relative timestamps
            t_norm = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-9)
            t_query = np.linspace(0, 1, config.CSI_FIXED_LEN)

            real_interp = np.stack([
                np.interp(t_query, t_norm, csi_clean[:, k].real)
                for k in range(config.N_SUBCARRIERS)
            ], axis=1)
            imag_interp = np.stack([
                np.interp(t_query, t_norm, csi_clean[:, k].imag)
                for k in range(config.N_SUBCARRIERS)
            ], axis=1)
            
            rx_complex = (real_interp + 1j * imag_interp).astype(np.complex64)
            
        receiver_stacks.append(rx_complex)

    if valid_receivers == 0:
        return None

    return np.stack(receiver_stacks, axis=0)


def _hampel_filter(data: np.ndarray, window_size: int, n_sigma: float) -> np.ndarray:
    """1D Hampel filter applied along the time axis (axis 1)."""
    med = ndimage.median_filter(data, size=(1, window_size, 1))
    mad = ndimage.median_filter(np.abs(data - med), size=(1, window_size, 1)) / 0.6745
    
    outliers = np.abs(data - med) > (n_sigma * mad)
    clean_data = np.copy(data)
    clean_data[outliers] = med[outliers]
    return clean_data


def _lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    """Butterworth lowpass filter applied along the time axis (axis 1)."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return data
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=1)


def _remove_dc(data: np.ndarray) -> np.ndarray:
    """Removes temporal DC component by subtracting the mean over time."""
    return data - np.mean(data, axis=1, keepdims=True)

def to_amplitude_phase(csi_window: np.ndarray) -> np.ndarray:
    """
    Applies the remaining time-domain filtering pipeline (Hampel, Lowpass,
    optional DC Removal) independently to Amplitude and Phase.
    """
    amp = np.abs(csi_window)
    # Unwrap phase smoothly across time now that it has been aligned to a uniform grid
    phase = np.unwrap(np.angle(csi_window), axis=1)

    # 1. Hampel Filter
    amp = _hampel_filter(amp, config.HAMPEL_WINDOW, config.HAMPEL_SIGMA)
    phase = _hampel_filter(phase, config.HAMPEL_WINDOW, config.HAMPEL_SIGMA)

    # 2. Lowpass Filter
    amp = _lowpass_filter(amp, config.LPF_CUTOFF_HZ, config.LPF_FS_HZ, config.LPF_ORDER)
    phase = _lowpass_filter(phase, config.LPF_CUTOFF_HZ, config.LPF_FS_HZ, config.LPF_ORDER)

    # 3. DC Removal (optional — see config.ENABLE_DC_REMOVAL). DC removal
    # subtracts each trial's own temporal mean, which discards exactly the
    # static path-loss/attenuation level a static weight-regression task
    # is plausibly most dependent on (as opposed to motion-sensing tasks,
    # where the DC term is nuisance and the AC/time-varying component is
    # the signal). An ablation study found disabling it improved CSI-only
    # test MAE substantially (~470g -> ~260g on that run) — default is
    # left True here to preserve prior pipeline behavior, but this is a
    # strong candidate to try disabled.
    if config.ENABLE_DC_REMOVAL:
        amp = _remove_dc(amp)
        phase = _remove_dc(phase)

    return np.stack([amp, phase], axis=-1).astype(np.float32)


if __name__ == "__main__":
    import os
    present = [p for p in config.CSI_FILES if os.path.exists(p)]
    if not present:
        print("No CSI files found next to this script — nothing to test.")
    else:
        df = load_csi_files(present)
        print(f"\nTotal packets after MAC filter: {len(df)}")
        print(f"Timestamp range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        print(f"Active Receivers: {df['receiver_id'].unique()}")

        import ledger
        led = ledger.load_ledger()
        row = led.iloc[0]
        window = extract_trial_csi(df, row["Timestamp_Start"], row["Timestamp_End"])
        
        if window is not None:
            print(f"\nSample trial '{row['sample_id']}': "
                  f"resampled CSI shape = {window.shape}, dtype = {window.dtype}")
            amp_phase = to_amplitude_phase(window)
            print(f"Final Pipeline shape = {amp_phase.shape}")
        else:
            print(f"\nSample trial '{row['sample_id']}' had completely invalid data.")