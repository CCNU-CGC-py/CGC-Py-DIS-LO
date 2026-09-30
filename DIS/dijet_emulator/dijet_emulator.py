"""
Dijet cross-section emulator (v3).

Predicts the dijet differential cross section sigma_T and sigma_L
from the physical parameters (x, Q, z, k1, q, theta).

The emulator was trained on inner tables built at fixed z=0.5, Q=2.0
(with eps = sqrt(Q^2 z(1-z)) as the free parameter).  The baked-in
polarization prefactors are divided out internally, so the user may
supply arbitrary (Q, z).

Usage:
    from dijet_emulator import dijet_cross_section, k1k2_to_k1qtheta

    # Option 1: native coordinates
    sT, sL = dijet_cross_section(x, Q, z, k1, q, theta)

    # Option 2: from (k1, k2, theta_12)
    k1, q, theta = k1k2_to_k1qtheta(k1, k2, theta_12)
    sT, sL = dijet_cross_section(x, Q, z, k1, q, theta)
"""

import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# ==========================================
# Model definition (inline to keep package self-contained)
# ==========================================
class ResidualBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.l1 = torch.nn.Linear(width, width)
        self.l2 = torch.nn.Linear(width, width)
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        h = torch.nn.functional.softplus(self.l1(x))
        h = self.l2(h)
        return x + self.alpha * h


class EmulatorNet(torch.nn.Module):
    def __init__(self, n_in=5, n_out=2, neck=128, n_res_blocks=12, expand=64, dropout=0.0):
        super().__init__()
        self.input_layer = torch.nn.Linear(n_in, expand)
        self.expand_layer = torch.nn.Linear(expand, neck)
        self.res_blocks = torch.nn.ModuleList([ResidualBlock(neck) for _ in range(n_res_blocks)])
        self.contract_layer = torch.nn.Linear(neck, expand)
        self.output_layer = torch.nn.Linear(expand, n_out)

    def forward(self, x):
        x = torch.nn.functional.softplus(self.input_layer(x))
        x = torch.nn.functional.softplus(self.expand_layer(x))
        for blk in self.res_blocks:
            x = torch.nn.functional.softplus(blk(x))
        x = torch.nn.functional.softplus(self.contract_layer(x))
        return self.output_layer(x)


# ==========================================
# Load model
# ==========================================
def _load_model(path_model=None):
    if path_model is None:
        path_model = os.path.join(_HERE, "model.pt")
    ckpt = torch.load(path_model, map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    model = EmulatorNet(
        n_in=mc["n_in"], n_out=mc["n_out"], neck=mc["neck"],
        n_res_blocks=mc["n_res_blocks"], expand=mc["expand"],
        dropout=mc.get("dropout", 0.0),
    )
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    scalers = ckpt["scalers"]
    return model, scalers


_model, _scalers = _load_model()

_x_mean = torch.tensor(_scalers["x_mean"], dtype=torch.float64, device=device)
_x_std = torch.tensor(_scalers["x_std"], dtype=torch.float64, device=device)
_y_mean = torch.tensor(_scalers["y_mean"], dtype=torch.float64, device=device)
_y_std = torch.tensor(_scalers["y_std"], dtype=torch.float64, device=device)


def _to_1d_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(device, dtype=torch.float64).view(-1)
    return torch.as_tensor(x, dtype=torch.float64, device=device).view(-1)


# ==========================================
# Coordinate transform
# ==========================================
def k1k2_to_k1qtheta(k1, k2, theta_12):
    """Convert (k1, k2, theta_12) to (k1, q, theta).

    Parameters
    ----------
    k1, k2 : array_like
        Jet transverse momenta.
    theta_12 : array_like
        Angle between k1 and k2.

    Returns
    -------
    k1, q, theta : ndarray
        k1 unchanged, q = |k1 + k2|, theta = angle between k1 and q.
    """
    k1 = np.asarray(k1, dtype=np.float64)
    k2 = np.asarray(k2, dtype=np.float64)
    theta_12 = np.asarray(theta_12, dtype=np.float64)

    q = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(theta_12))
    qx = k1 + k2 * np.cos(theta_12)
    qy = k2 * np.sin(theta_12)
    theta = np.arctan2(qy, qx)

    return k1, q, theta


# ==========================================
# Baked-in prefactors (z=0.5, Q=2.0 in the inner table build)
# ==========================================
_BAKED_Z = 0.5
_BAKED_Q = 2.0
_BAKED_PREF_T = (1.0 - _BAKED_Z) ** 2 + _BAKED_Z ** 2          # 0.5
_BAKED_PREF_L = 4.0 * _BAKED_Q**2 * _BAKED_Z**2 * (1.0 - _BAKED_Z)**2  # 1.0


# ==========================================
# Cross section
# ==========================================
def dijet_cross_section(x, Q, z, k1, q, theta, grid=None):
    """Predict the dijet cross section for given (x, Q, z, k1, q, theta).

    The emulator was trained at fixed z=0.5, Q=2.0 with eps as the free
    parameter.  The baked-in polarization prefactors are divided out and
    replaced by the correct ones for the user's (Q, z).

    Parameters
    ----------
    x : array_like
        Bjorken-x.
    Q : array_like
        Photon virtuality.
    z : array_like
        Quark longitudinal momentum fraction.
    k1 : array_like
        Jet transverse momentum.
    q : array_like
        |k1 + k2| magnitude.
    theta : array_like
        Angle between k1 and q.
    grid : bool or None
        If True, compute on the outer product of all distinct input arrays.
        If None, auto-detect (grid=True when lengths differ).

    Returns
    -------
    sigma_T, sigma_L : ndarray
        Transverse and longitudinal cross sections (without overall
        constants alpha_em, e_q^2, N_c, S_perp, 2/(2pi)^6).
    """
    is_numpy = any(isinstance(v, np.ndarray) for v in [x, Q, z, k1, q, theta])

    x_t = _to_1d_tensor(x)
    Q_t = _to_1d_tensor(Q)
    z_t = _to_1d_tensor(z)
    k1_t = _to_1d_tensor(k1)
    q_t = _to_1d_tensor(q)
    theta_t = _to_1d_tensor(theta)

    # eps = sqrt(Q^2 z (1-z))
    eps_t = torch.sqrt(Q_t * Q_t * z_t * (1.0 - z_t))

    lengths = [len(x_t), len(eps_t), len(k1_t), len(q_t), len(theta_t)]
    if grid is None:
        grid = len(set(lengths)) > 1

    if grid:
        grids = torch.meshgrid(x_t, eps_t, k1_t, q_t, theta_t, indexing='ij')
        shape_out = grids[0].shape
        flat = torch.stack([g.reshape(-1) for g in grids], dim=1)
    else:
        flat = torch.stack(
            torch.broadcast_tensors(x_t, eps_t, k1_t, q_t, theta_t), dim=1
        )
        shape_out = flat.shape[0]

    # standardize: log10 for (x, eps, k1, q), linear for theta
    inp = flat.clone()
    inp[:, :4] = torch.log10(inp[:, :4].clamp(min=1e-30))
    inp = ((inp - _x_mean) / _x_std).float()

    with torch.no_grad():
        pred_std = _model(inp)

    # inverse: unstandardize then 10^(...)
    log_phys = pred_std.double() * _y_std + _y_mean
    phys = torch.pow(10.0, log_phys)

    # Raw emulator output has baked-in prefactors at z=0.5, Q=2.0.
    # Divide those out to get the bare kernel, then apply correct (Q, z).
    raw_T = phys[:, 0] / _BAKED_PREF_T
    raw_L = phys[:, 1] / _BAKED_PREF_L

    if grid:
        raw_T = raw_T.view(shape_out)
        raw_L = raw_L.view(shape_out)
        ndim = len(shape_out)
        bcast = [1] * ndim
        bcast[1] = -1
        pref_T = ((1.0 - z_t)**2 + z_t**2).view(bcast)
        pref_L = (4.0 * Q_t**2 * z_t**2 * (1.0 - z_t)**2).view(bcast)
    else:
        pref_T = (1.0 - z_t)**2 + z_t**2
        pref_L = 4.0 * Q_t**2 * z_t**2 * (1.0 - z_t)**2

    _TWOPI2 = (2.0 * np.pi) ** 2
    sigma_T = _TWOPI2 * pref_T * raw_T
    sigma_L = _TWOPI2 * pref_L * raw_L

    if is_numpy:
        return sigma_T.cpu().numpy(), sigma_L.cpu().numpy()
    return sigma_T, sigma_L
