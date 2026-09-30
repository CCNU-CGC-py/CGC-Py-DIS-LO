"""Generate the three-dimensional Sudakov-evolved WW TMD table.

The radial Hankel transform uses composite Gauss-Legendre panels. Logarithmic
panel edges resolve the coordinate-space WW profile, while linear edges with
spacing pi/k resolve the oscillations of J_0(k r). The finite running-coupling
integral is also evaluated with Gauss-Legendre quadrature after changing the
integration variable to log(mu^2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.special import j0, roots_legendre


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_IMPORT_CWD = Path.cwd()
try:
    # dipole_function.py currently resolves its checkpoint relative to cwd.
    os.chdir(PROJECT_ROOT)
    from dipole_function import ww_alpha_s_f_r_autograd  # noqa: E402
finally:
    os.chdir(_IMPORT_CWD)


TABLE_MAGIC = b"FWW3D01\0"


@dataclass(frozen=True)
class FWWTableConfig:
    """Numerical and physical settings used to generate the table."""

    x_min: float = 1.0e-7
    x_max: float = 1.0e-2
    n_x: int = 100
    k_min: float = 1.0e-3
    k_max: float = 1.0e2
    n_k: int = 100
    q2_max: float = 400.0
    n_q2: int = 200
    g2: float = 0.225
    nc: float = 3.0
    s_perp: float = 1.0
    lambda_qcd: float = 0.2
    n_f: int = 3
    alpha_max: float = 0.7
    x0_for_y: float = 0.03
    eps: float = 1.0e-10
    n_mu_gauss: int = 64
    radial_gauss_order: int = 16
    radial_log_step: float = 0.05
    radial_min: float = 1.0e-9
    radial_hard_max: float = 3000.0
    radial_tail_min: float = 3000.0
    radial_low_k_cycles: float = 10.0
    radial_damping_tolerance: float = 1.0e-14
    profile_points: int = 32768
    q2_chunk: int = 32

    @property
    def c_f(self) -> float:
        return (self.nc**2 - 1.0) / (2.0 * self.nc)

    @property
    def c_a(self) -> float:
        return self.nc


def load_qs2_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the two-column x, Q_s^2 table."""

    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"Invalid Q_s^2 table: {path}")
    x = np.asarray(data[:, 0], dtype=np.float64)
    qs2 = np.asarray(data[:, 1], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    qs2 = qs2[order]
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(qs2)):
        raise ValueError("The Q_s^2 table contains non-finite values.")
    if np.any(x <= 0.0) or np.any(qs2 <= 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("The Q_s^2 table axes must be positive and strictly increasing.")
    return x, qs2


def interpolate_qs2(
    x: np.ndarray | float,
    table_x: np.ndarray,
    table_qs2: np.ndarray,
) -> np.ndarray:
    """Interpolate Q_s^2 linearly in log(x), without extrapolation."""

    values = np.asarray(x, dtype=np.float64)
    if np.any(values < table_x[0]) or np.any(values > table_x[-1]):
        raise ValueError("Requested x lies outside the Q_s^2 table.")
    return np.interp(np.log(values), np.log(table_x), table_qs2)


def build_table_grids(
    config: FWWTableConfig,
    table_x: np.ndarray,
    table_qs2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build logarithmic x, k, and per-x Q^2 grids."""

    x_grid = np.geomspace(config.x_min, config.x_max, config.n_x, dtype=np.float64)
    k_grid = np.geomspace(config.k_min, config.k_max, config.n_k, dtype=np.float64)
    qs2_grid = interpolate_qs2(x_grid, table_x, table_qs2)
    eta_grid = np.linspace(0.0, 1.0, config.n_q2, dtype=np.float64)
    q2_grid = qs2_grid[:, None] * (
        config.q2_max / qs2_grid[:, None]
    ) ** eta_grid[None, :]
    return x_grid, k_grid, qs2_grid, eta_grid, q2_grid


def alpha_s_frozen_one_loop(
    mu2: np.ndarray | float,
    config: FWWTableConfig,
) -> np.ndarray:
    """Evaluate the frozen one-loop coupling used in the notebook."""

    values = np.asarray(mu2, dtype=np.float64)
    if np.any(values <= 0.0):
        raise ValueError("All mu^2 values must be positive.")
    beta0 = 11.0 - 2.0 * config.n_f / 3.0
    lambda2 = config.lambda_qcd**2
    freeze_mu2 = lambda2 * np.exp(4.0 * np.pi / (beta0 * config.alpha_max))
    effective = np.maximum(values, freeze_mu2)
    return 4.0 * np.pi / (beta0 * np.log(effective / lambda2))


def sudakov_perturbative_gauss_legendre(
    qs2: float,
    q2: np.ndarray,
    config: FWWTableConfig,
    order: int | None = None,
) -> np.ndarray:
    r"""Evaluate the r-independent running-coupling Sudakov contribution.

    The variable t = log(mu^2) gives

        C_A/(2 pi) int_log(qs2)^log(q2) dt alpha_s(exp(t)) [log(q2)-t].
    """

    q2_values = np.asarray(q2, dtype=np.float64)
    if qs2 <= 0.0 or np.any(q2_values < qs2) or np.any(~np.isfinite(q2_values)):
        raise ValueError("Require finite Q^2 >= Q_s^2 > 0.")
    n = config.n_mu_gauss if order is None else int(order)
    z, weight = roots_legendre(n)
    logarithmic_span = np.log(q2_values / qs2)
    log_qs2 = math.log(qs2)
    log_mu2 = log_qs2 + 0.5 * logarithmic_span[:, None] * (z[None, :] + 1.0)
    log_q2 = np.log(q2_values)[:, None]
    integrand = alpha_s_frozen_one_loop(np.exp(log_mu2), config) * (log_q2 - log_mu2)
    integral = 0.5 * logarithmic_span * np.sum(weight[None, :] * integrand, axis=1)
    return config.c_a / (2.0 * np.pi) * integral


def radial_upper_limit(k: float, config: FWWTableConfig) -> float:
    """Choose a validated finite proxy for the Q^2=Q_s^2 radial tail."""

    return min(
        config.radial_hard_max,
        max(config.radial_tail_min, config.radial_low_k_cycles / k),
    )


def composite_radial_gauss_legendre(
    k: float,
    r_max: float,
    config: FWWTableConfig,
    order: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return nodes and weights for int_0^r_max dr using composite panels.

    The union of logarithmic edges and edges spaced by pi/k resolves both the
    WW profile and every sign-changing Bessel oscillation. Each panel uses an
    independent Gauss-Legendre rule.
    """

    if not np.isfinite(k) or not np.isfinite(r_max) or k <= 0.0 or r_max <= 0.0:
        raise ValueError("Require finite k > 0 and r_max > 0.")
    n = config.radial_gauss_order if order is None else int(order)
    z, weight = roots_legendre(n)

    n_log = max(
        1,
        int(math.ceil(math.log(r_max / config.radial_min) / config.radial_log_step)),
    )
    log_edges = np.geomspace(config.radial_min, r_max, n_log + 1)
    n_osc = max(1, int(math.ceil(k * r_max / np.pi)))
    oscillation_edges = np.linspace(0.0, r_max, n_osc + 1)
    edges = np.unique(np.concatenate(([0.0], log_edges, oscillation_edges, [r_max])))

    left = edges[:-1]
    right = edges[1:]
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    nodes = (midpoint[:, None] + half_width[:, None] * z[None, :]).reshape(-1)
    weights = (half_width[:, None] * weight[None, :]).reshape(-1)
    return nodes, weights


def coordinate_ww_profile(
    x: float,
    qs2: float,
    config: FWWTableConfig,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Tabulate the coordinate input [alpha_s F_WW]/alpha_s(Q_s^2)."""

    r_profile = np.geomspace(
        config.radial_min,
        config.radial_hard_max,
        config.profile_points,
        dtype=np.float64,
    )
    alpha_s_fww = ww_alpha_s_f_r_autograd(
        r_profile,
        x,
        S_perp=config.s_perp,
        Nc=config.nc,
        x0_for_Y=config.x0_for_y,
        eps=config.eps,
        dtype=torch.float64,
    )
    q_tilde = np.asarray(alpha_s_fww, dtype=np.float64) / float(
        alpha_s_frozen_one_loop(qs2, config)
    )
    if np.any(~np.isfinite(q_tilde)):
        raise FloatingPointError(f"Non-finite coordinate WW profile at x={x:.8e}.")
    return r_profile, q_tilde


def _interpolate_profile(
    radial_nodes: np.ndarray,
    profile_r: np.ndarray,
    profile_values: np.ndarray,
) -> np.ndarray:
    """Interpolate the smooth WW profile linearly in log(r)."""

    safe_nodes = np.maximum(radial_nodes, profile_r[0])
    return np.interp(
        np.log(safe_nodes),
        np.log(profile_r),
        profile_values,
        left=profile_values[0],
        right=0.0,
    )


def evaluate_x_slice(
    x: float,
    qs2: float,
    k_grid: np.ndarray,
    q2_grid: np.ndarray,
    config: FWWTableConfig,
    device: torch.device | str | None = None,
) -> np.ndarray:
    r"""Evaluate F_WW(x,k,Q^2) for one complete k-Q^2 slice."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    profile_r, profile_values = coordinate_ww_profile(x, qs2, config)
    q2_values = np.asarray(q2_grid, dtype=np.float64)
    logarithmic_span = np.log(q2_values / qs2)
    sudakov_pert = sudakov_perturbative_gauss_legendre(qs2, q2_values, config)
    output = np.empty((len(k_grid), len(q2_values)), dtype=np.float64)

    for ik, k in enumerate(np.asarray(k_grid, dtype=np.float64)):
        r_max_zero = radial_upper_limit(float(k), config)
        nodes, weights = composite_radial_gauss_legendre(float(k), r_max_zero, config)
        profile = _interpolate_profile(nodes, profile_r, profile_values)
        base = weights * nodes * j0(k * nodes) * profile
        output[ik, 0] = np.sum(base, dtype=np.float64) / (2.0 * np.pi)

        if len(q2_values) == 1:
            continue

        first_positive_span = logarithmic_span[1]
        if first_positive_span <= 0.0:
            raise ValueError("The Q^2 grid must be strictly increasing after Q_s^2.")
        damping_limit = math.sqrt(
            -math.log(config.radial_damping_tolerance)
            / (config.g2 * first_positive_span)
        )
        r_max_damped = min(r_max_zero, damping_limit)
        nodes, weights = composite_radial_gauss_legendre(float(k), r_max_damped, config)
        profile = _interpolate_profile(nodes, profile_r, profile_values)
        base = weights * nodes * j0(k * nodes) * profile / (2.0 * np.pi)

        nodes2_t = torch.as_tensor(nodes * nodes, dtype=torch.float64, device=device)
        base_t = torch.as_tensor(base, dtype=torch.float64, device=device)
        for start in range(1, len(q2_values), config.q2_chunk):
            stop = min(start + config.q2_chunk, len(q2_values))
            span_t = torch.as_tensor(
                logarithmic_span[start:stop], dtype=torch.float64, device=device
            )
            damped = torch.exp(-config.g2 * span_t[:, None] * nodes2_t[None, :])
            radial = torch.matmul(damped, base_t)
            output[ik, start:stop] = (
                torch.exp(
                    -torch.as_tensor(
                        sudakov_pert[start:stop], dtype=torch.float64, device=device
                    )
                )
                * radial
            ).detach().cpu().numpy()

    if np.any(~np.isfinite(output)):
        raise FloatingPointError(f"Non-finite momentum-space WW values at x={x:.8e}.")
    return output


def write_binary_table(
    path: str | Path,
    x_grid: np.ndarray,
    k_grid: np.ndarray,
    qs2_grid: np.ndarray,
    eta_grid: np.ndarray,
    values: np.ndarray,
    config: FWWTableConfig,
) -> Path:
    """Write the compact, versioned binary format consumed by C++."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_shape = (len(x_grid), len(k_grid), len(eta_grid))
    if values.shape != expected_shape:
        raise ValueError(f"Expected values with shape {expected_shape}, got {values.shape}.")

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        stream.write(TABLE_MAGIC)
        stream.write(struct.pack("<QQQ", *expected_shape))
        stream.write(struct.pack("<dd", config.q2_max, config.g2))
        for array in (x_grid, k_grid, qs2_grid, eta_grid, values):
            contiguous = np.ascontiguousarray(array, dtype="<f8")
            stream.write(contiguous.tobytes(order="C"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(output_path)
    return output_path


def read_binary_table(
    path: str | Path,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the C++ table format for Python-side validation and plotting."""

    with Path(path).open("rb") as stream:
        magic = stream.read(8)
        if magic != TABLE_MAGIC:
            raise ValueError(f"Unexpected table magic {magic!r}.")
        n_x, n_k, n_q2 = struct.unpack("<QQQ", stream.read(24))
        q2_max, g2 = struct.unpack("<dd", stream.read(16))
        x_grid = np.fromfile(stream, dtype="<f8", count=n_x)
        k_grid = np.fromfile(stream, dtype="<f8", count=n_k)
        qs2_grid = np.fromfile(stream, dtype="<f8", count=n_x)
        eta_grid = np.fromfile(stream, dtype="<f8", count=n_q2)
        values = np.fromfile(stream, dtype="<f8", count=n_x * n_k * n_q2)
        if values.size != n_x * n_k * n_q2 or stream.read(1):
            raise ValueError("The binary table has an unexpected size.")
    metadata = {
        "n_x": int(n_x),
        "n_k": int(n_k),
        "n_q2": int(n_q2),
        "q2_max": q2_max,
        "g2": g2,
    }
    return metadata, x_grid, k_grid, qs2_grid, eta_grid, values.reshape(n_x, n_k, n_q2)


def generate_table(
    config: FWWTableConfig,
    qs2_table_path: str | Path,
    work_directory: str | Path,
    output_path: str | Path,
    device: torch.device | str | None = None,
) -> Path:
    """Generate all x slices with a resumable NumPy memmap checkpoint."""

    table_x, table_qs2 = load_qs2_table(qs2_table_path)
    x_grid, k_grid, qs2_grid, eta_grid, q2_grid = build_table_grids(
        config, table_x, table_qs2
    )

    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    values_path = work / "fww_values.npy"
    progress_path = work / "progress.json"
    config_path = work / "config.json"
    config_payload = asdict(config)

    if config_path.exists():
        stored_config = json.loads(config_path.read_text(encoding="utf-8"))
        if stored_config != config_payload:
            raise RuntimeError(
                f"Existing work directory {work} uses a different configuration."
            )
    else:
        config_path.write_text(
            json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    shape = (config.n_x, config.n_k, config.n_q2)
    if values_path.exists():
        values = np.lib.format.open_memmap(values_path, mode="r+")
        if values.shape != shape or values.dtype != np.float64:
            raise RuntimeError("Existing checkpoint array has the wrong shape or dtype.")
    else:
        values = np.lib.format.open_memmap(
            values_path, mode="w+", dtype=np.float64, shape=shape
        )
        values[:] = np.nan
        values.flush()

    completed: set[int] = set()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = {int(index) for index in progress.get("completed_x_indices", [])}

    start_time = time.perf_counter()
    for ix, (x, qs2, q2_values) in enumerate(zip(x_grid, qs2_grid, q2_grid)):
        if ix in completed and np.all(np.isfinite(values[ix])):
            print(f"[{ix + 1:3d}/{config.n_x}] x={x:.6e}: checkpoint already complete")
            continue
        slice_start = time.perf_counter()
        values[ix] = evaluate_x_slice(
            float(x), float(qs2), k_grid, q2_values, config, device=device
        )
        values.flush()
        completed.add(ix)
        progress_path.write_text(
            json.dumps(
                {
                    "completed_x_indices": sorted(completed),
                    "last_x": float(x),
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(
            f"[{ix + 1:3d}/{config.n_x}] x={x:.6e}: "
            f"{time.perf_counter() - slice_start:.2f} s"
        )

    if np.any(~np.isfinite(values)):
        raise FloatingPointError("The completed table still contains non-finite values.")
    path = write_binary_table(
        output_path, x_grid, k_grid, qs2_grid, eta_grid, np.asarray(values), config
    )
    print(f"Wrote {path} ({path.stat().st_size / 1024**2:.2f} MiB)")
    return path


def _parse_arguments(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qs2-table",
        type=Path,
        default=PROJECT_ROOT / "DIS" / "src" / "Qs2_table.dat",
    )
    parser.add_argument(
        "--work-directory",
        type=Path,
        default=PROJECT_ROOT / "fww_table_work",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "FWW_TMD_TABLE.bin",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = _parse_arguments(arguments)
    config = FWWTableConfig()
    generate_table(
        config,
        qs2_table_path=args.qs2_table,
        work_directory=args.work_directory,
        output_path=args.output,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
