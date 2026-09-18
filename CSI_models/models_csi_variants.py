"""
CSI-only model architectures beyond the plain CNN (CSIRegressor in
models.py): LSTM, RNN, Transformer, and a VAE-based regressor.

All four consume the same (B, n_rx, T, n_subcarriers, 2) input as
CSIRegressor — the same CSIOnlyDataset, same CSIScaler-normalized
features — so a comparison against the CNN baseline is apples-to-apples;
only the architecture differs.

LSTM/RNN/Transformer reshape the input into a (B, T, n_rx*n_sub*2)
sequence (each time step's flattened receiver+subcarrier+[amplitude,phase]
values as that step's feature vector) and pool over time for the final
regression head. The VAE instead treats it as a (B, n_rx*2, T, n_sub)
"image" (matching CSIEncoder's convention in models.py), so it can use a
convolutional encoder/decoder for reconstruction.
"""

import math

import torch
import torch.nn as nn

import config
from models import RegressionHead


def _to_sequence(x):
    """(B, n_rx, T, n_sub, 2) -> (B, T, n_rx*n_sub*2)"""
    b, n_rx, t, n_sub, ch = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b, t, n_rx * n_sub * ch)


# ---------------------------------------------------------------------------
# LSTM / RNN
# ---------------------------------------------------------------------------
class CSILSTMRegressor(nn.Module):
    def __init__(self, n_rx, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        input_dim = n_rx * config.N_SUBCARRIERS * 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = RegressionHead(hidden_dim, dropout=dropout)

    def forward(self, x):
        seq = _to_sequence(x)
        out, (h, c) = self.lstm(seq)
        last_step = out[:, -1, :]     # final time step's hidden output
        return self.head(last_step)


class CSIRNNRegressor(nn.Module):
    """Vanilla (Elman) RNN — included as the classic recurrent baseline
    alongside LSTM, despite its known vanishing-gradient limitations on
    longer sequences (T=100 here)."""

    def __init__(self, n_rx, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        input_dim = n_rx * config.N_SUBCARRIERS * 2
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           nonlinearity="tanh", dropout=dropout if num_layers > 1 else 0.0)
        self.head = RegressionHead(hidden_dim, dropout=dropout)

    def forward(self, x):
        seq = _to_sequence(x)
        out, h = self.rnn(seq)
        last_step = out[:, -1, :]
        return self.head(last_step)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (no extra learned parameters)."""

    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CSITransformerRegressor(nn.Module):
    """
    Small encoder-only Transformer (kept deliberately small — d_model=64,
    2 layers — given ~45 training samples; a larger Transformer would
    almost certainly overfit before it out-learns that constraint).
    Mean-pools over the time dimension before the regression head, rather
    than a CLS-token, to keep the architecture simple.
    """

    def __init__(self, n_rx, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        input_dim = n_rx * config.N_SUBCARRIERS * 2
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = RegressionHead(d_model, dropout=dropout)

    def forward(self, x):
        seq = _to_sequence(x)
        h = self.input_proj(seq)
        h = self.pos_enc(h)
        h = self.transformer(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------
class CSIVAERegressor(nn.Module):
    """
    Convolutional VAE over the CSI "image" (n_rx*2 channels x T x
    n_subcarriers, same convention as CSIEncoder in models.py), trained
    with a combined loss (regression + reconstruction + KL) — see
    evaluate_csi_models.py's train_vae_epoch for the training loop, since
    this compound loss doesn't fit train_utils.run_epoch's single-
    criterion signature used by the other models.

    The regression head reads from `mu` (the latent mean), not a sampled
    `z` — deterministic at both train and eval time. Sampling still
    happens (via reparameterize) for the reconstruction path, so the KL
    term has something to regularize.
    """

    def __init__(self, n_rx, latent_dim=32, dropout=0.3):
        super().__init__()
        self.in_ch = n_rx * 2
        T, S = config.CSI_FIXED_LEN, config.N_SUBCARRIERS
        if T % 4 != 0 or S % 4 != 0:
            raise ValueError(f"CSIVAERegressor requires CSI_FIXED_LEN and N_SUBCARRIERS "
                              f"divisible by 4 for exact encoder/decoder size matching "
                              f"(got T={T}, S={S}).")
        self.flat_t, self.flat_s = T // 4, S // 4

        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        flat_dim = 64 * self.flat_t * self.flat_s
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, self.in_ch, kernel_size=2, stride=2),
        )
        self.head = RegressionHead(latent_dim, dropout=dropout)

    def to_image(self, x):
        """(B, n_rx, T, n_sub, 2) -> (B, n_rx*2, T, n_sub)"""
        b, n_rx, t, n_sub, ch = x.shape
        return x.permute(0, 1, 4, 2, 3).reshape(b, n_rx * ch, t, n_sub)

    def encode(self, x):
        img = self.to_image(x)
        h = self.enc_conv(img).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 64, self.flat_t, self.flat_s)
        return self.dec_conv(h)

    def forward(self, x):
        """Regression-only path, for compatibility with the shared
        evaluate_in_grams forward_fn pattern used by every other model."""
        mu, _ = self.encode(x)
        return self.head(mu)
