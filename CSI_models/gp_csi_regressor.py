"""
Gaussian Process regression on CSI, with RBF and Matern kernels.

Unlike the neural models, a GP is NOT run on the raw (n_rx, T, n_sub, 2)
tensor flattened to ~38,400 dimensions. With only ~45 training samples,
a kernel computed over that many raw dimensions would be dominated by
curse-of-dimensionality noise (distances between points concentrate,
so the kernel stops encoding meaningful similarity) rather than genuine
structure — this is a real failure mode for kernel methods in high
dimensions with few samples, not a minor detail. Standard practice for
classical/kernel-method CSI featurization is to collapse the time axis
into summary statistics per (receiver, subcarrier, channel) first.

extract_gp_features() does exactly that: mean and std over time, giving
a flat feature vector of size n_rx * n_subcarriers * 2[amp,phase] *
2[mean,std]. This is a real, deliberate difference from how the other
models see the data, not an oversight — flagged here and again in
evaluate_csi_models.py's summary output.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


def extract_gp_features(csi_amp_phase: np.ndarray) -> np.ndarray:
    """csi_amp_phase: (n_rx, T, n_sub, 2) -> flat 1D feature vector, length n_rx*n_sub*2*2."""
    mean_feat = csi_amp_phase.mean(axis=1)   # (n_rx, n_sub, 2)
    std_feat = csi_amp_phase.std(axis=1)     # (n_rx, n_sub, 2)
    return np.concatenate([mean_feat.flatten(), std_feat.flatten()]).astype(np.float64)


def build_kernel(name: str):
    if name == "rbf":
        return (ConstantKernel(1.0, (1e-2, 1e2))
                * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
                + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2)))
    if name == "matern":
        return (ConstantKernel(1.0, (1e-2, 1e2))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
                + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2)))
    raise ValueError(f"Unknown GP kernel name: {name!r} (expected 'rbf' or 'matern')")


def fit_and_predict(kernel_name, X_train, y_train, X_test, seed, n_restarts=5):
    """
    Fits a GaussianProcessRegressor (features standardized via a
    StandardScaler fit on X_train only) and returns (pred, std) on
    X_test, both in the same (already target-scaled) units y_train was
    given in — inverse-transform pred with WeightScaler as usual;
    std needs scaling by WeightScaler.std ONLY (not the mean offset —
    a standard deviation is not a location, so WeightScaler.inverse_transform
    would incorrectly shift it) to convert to real grams.
    """
    scaler = StandardScaler().fit(X_train)
    kernel = build_kernel(kernel_name)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts,
                                   random_state=seed, normalize_y=False)
    gp.fit(scaler.transform(X_train), y_train)
    pred, std = gp.predict(scaler.transform(X_test), return_std=True)
    return pred, std, gp
