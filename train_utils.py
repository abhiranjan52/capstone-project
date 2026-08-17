"""
Shared training utilities: early stopping, regression metrics (computed in
real grams, not the standardized target scale), and generic train/eval
epoch loops reused by train_csi.py, train_video.py, and train_fusion.py.
"""

import numpy as np
import torch
import torch.nn as nn


def get_device():
    import config_train
    if config_train.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if `value` is the new best."""
        is_better = (
            self.best is None
            or (self.mode == "min" and value < self.best - self.min_delta)
            or (self.mode == "max" and value > self.best + self.min_delta)
        )
        if is_better:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def regression_metrics(preds_g: np.ndarray, targets_g: np.ndarray) -> dict:
    """MAE / RMSE / R^2, all in real grams."""
    errors = preds_g - targets_g
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets_g - targets_g.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")
    return {"mae_g": mae, "rmse_g": rmse, "r2": r2, "n": len(targets_g)}


def save_checkpoint(path, model_state, extra: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model_state}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


def run_epoch(model, loader, criterion, device, optimizer=None, forward_fn=None):
    """
    One epoch over `loader`. If `optimizer` is given, trains (backward +
    step); otherwise runs in eval mode with no_grad.

    `forward_fn(model, batch, device) -> (pred_scaled, target_scaled, weight_g, occluded)`
    lets each stage supply its own batch-unpacking logic while sharing
    this loop.

    Returns: avg_loss, regression_metrics_dict (in real grams)
    """
    import config_train
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    n_batches = 0
    all_preds_g, all_targets_g = [], []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            pred_scaled, target_scaled, weight_g, occluded = forward_fn(model, batch, device)
            loss = criterion(pred_scaled, target_scaled)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config_train.GRAD_CLIP_NORM,
                )
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_preds_g.append(pred_scaled.detach().cpu().numpy())
            all_targets_g.append(target_scaled.detach().cpu().numpy())

    # Metrics are computed by the caller after inverse-scaling — this loop
    # only returns the scaled arrays plus the running loss to keep the
    # scaler out of this generic function.
    return total_loss / max(n_batches, 1), np.concatenate(all_preds_g), np.concatenate(all_targets_g)


def evaluate_in_grams(model, loader, device, forward_fn, scaler, occluded_filter=None):
    """
    Full evaluation pass, inverse-scaled to real grams, optionally sliced
    by `occluded_filter` (True/False/None for all).
    Returns regression_metrics dict.
    """
    model.eval()
    all_preds_g, all_targets_g, all_occluded = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred_scaled, target_scaled, weight_g, occluded = forward_fn(model, batch, device)
            preds_g = scaler.inverse_transform(pred_scaled.detach().cpu().numpy())
            all_preds_g.append(preds_g)
            all_targets_g.append(np.asarray(weight_g, dtype=np.float32))
            all_occluded.append(np.asarray(occluded, dtype=bool))

    preds_g = np.concatenate(all_preds_g)
    targets_g = np.concatenate(all_targets_g)
    occluded = np.concatenate(all_occluded)

    if occluded_filter is not None:
        mask = occluded == occluded_filter
        preds_g, targets_g = preds_g[mask], targets_g[mask]

    if len(targets_g) == 0:
        return {"mae_g": float("nan"), "rmse_g": float("nan"), "r2": float("nan"), "n": 0}
    return regression_metrics(preds_g, targets_g)
