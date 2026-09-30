"""Callable elastic and inelastic DIS cross sections.

The public functions accept either scalars or NumPy arrays.  All non-scalar
arrays in one call must have exactly the same shape; scalar arguments are
expanded to that shape.  Polarization ``0`` selects the transverse component
and polarization ``1`` selects the longitudinal component.

The normalization and hard factors follow ``new.ipynb``.  Quark flavor is
specified by a PDG ID, and the target-area default is ``S_perp = 1``.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
import sys
from typing import Callable, Iterator, Mapping

import numpy as np
from numpy.polynomial.legendre import leggauss


NC = 3.0
ALPHA_EM = 1.0 / 137.0
S_PERP = 1.0
X0_DIPOLE = 0.03
# The executed integrate_emulator_TL cell in new.ipynb uses (2*pi)^(-4).
# INELASTIC_FOURIER_POWER = 6

# Keep the validated high-accuracy default, while allowing large event-level
# analyses to opt into a separately convergence-tested quadrature order.
DEFAULT_N_RADIAL = int(os.environ.get("CROSS_SECTION_N_RADIAL", "192"))
if DEFAULT_N_RADIAL <= 0:
    raise ValueError("CROSS_SECTION_N_RADIAL must be a positive integer.")
DEFAULT_ELL_MAX = 100.0
DEFAULT_ELL_SCALE = 1.0

_DIPOLE_CALL_BATCH_SIZE = int(
    os.environ.get("CROSS_SECTION_DIPOLE_BATCH_SIZE", "384")
)
if _DIPOLE_CALL_BATCH_SIZE <= 0:
    raise ValueError(
        "CROSS_SECTION_DIPOLE_BATCH_SIZE must be a positive integer."
    )
_DIPOLE_XG_BLOCK_SIZE = 32
_ELASTIC_EVENT_BATCH_SIZE = 1024
_EMULATOR_BATCH_SIZE = 50_000

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_DIPOLE_MODULE_FILE = _PROJECT_ROOT / "dipole_function.py"
DIJET_EMULATOR_SOURCE = _HERE / "dijet_emulator" / "dijet_emulator.py"
DIJET_EMULATOR_CHECKPOINT = _HERE / "dijet_emulator" / "model.pt"

__all__ = [
    "dipole",
    "elastic_dipole_device",
    "quark_charge_squared",
    "el_cross_section",
    "elastic_cross_section",
    "inel_cross_section",
    "inelastic_cross_section",
    "inelastic_emulator_backend_id",
    "inelastic_emulator_device",
    "DIJET_EMULATOR_SOURCE",
    "DIJET_EMULATOR_CHECKPOINT",
]


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily change directory while importing a legacy local module."""
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


@lru_cache(maxsize=1)
def _project_dipole_function() -> Callable[..., object]:
    """Load PINN_dipole_k from the current project dipole implementation."""
    if not _DIPOLE_MODULE_FILE.is_file():
        raise FileNotFoundError(
            f"The dipole implementation does not exist: {_DIPOLE_MODULE_FILE}"
        )

    project_root = str(_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    loaded = sys.modules.get("dipole_function")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        if loaded_file != _DIPOLE_MODULE_FILE.resolve():
            raise ImportError(
                "A different module named 'dipole_function' is already loaded: "
                f"{loaded_file}"
            )

    # The legacy module loads model_ciBK_best.pt relative to the current
    # directory, so only its first import is performed from the project root.
    with _working_directory(_PROJECT_ROOT):
        module = importlib.import_module("dipole_function")

    function = getattr(module, "PINN_dipole_k", None)
    if function is None:
        raise ImportError(
            f"PINN_dipole_k was not found in {_DIPOLE_MODULE_FILE}."
        )
    return function


def dipole(ell: object, xg: object) -> float | np.ndarray:
    """Evaluate the current momentum-space dipole at pairwise ``(ell, xg)``.

    This is the only adapter that depends on the concrete dipole model.  To
    replace the dipole implementation later, change this function while
    retaining its pairwise scalar/array interface.
    """
    ell_array, xg_array = np.broadcast_arrays(
        np.asarray(ell, dtype=np.float64),
        np.asarray(xg, dtype=np.float64),
    )
    scalar_input = ell_array.ndim == 0
    if np.any(~np.isfinite(ell_array)) or np.any(ell_array <= 0.0):
        raise ValueError("ell must contain finite, strictly positive values.")
    if np.any(~np.isfinite(xg_array)) or np.any(
        (xg_array <= 0.0) | (xg_array > X0_DIPOLE)
    ):
        raise ValueError(
            f"xg must contain finite values satisfying 0 < xg <= {X0_DIPOLE}."
        )

    rapidity = np.log(X0_DIPOLE / xg_array)
    # The project backend converts these arrays to PyTorch tensors.  Explicit
    # copies avoid passing read-only views created by NumPy broadcasting.
    ell_flat = ell_array.reshape(-1).copy()
    rapidity_flat = rapidity.reshape(-1).copy()
    values = _project_dipole_function()(
        ell_flat,
        rapidity_flat,
        representation="fundamental",
        grid=False,
    )
    result = np.asarray(values, dtype=np.float64).reshape(ell_array.shape)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("The dipole model returned a non-finite value.")
    return float(result) if scalar_input else result


def elastic_dipole_device() -> str:
    """Return the PyTorch device selected by the dipole backend."""
    _project_dipole_function()
    module = sys.modules["dipole_function"]
    return str(module.device)


@lru_cache(maxsize=1)
def _dijet_emulator_module() -> object:
    """Load the legacy dijet emulator and its adjacent checkpoint."""
    for path in (DIJET_EMULATOR_SOURCE, DIJET_EMULATOR_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"Dijet emulator file not found: {path}")

    module_path = str(_HERE)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    from dijet_emulator import dijet_emulator

    loaded_file = Path(dijet_emulator.__file__).resolve()
    if loaded_file != DIJET_EMULATOR_SOURCE.resolve():
        raise ImportError(
            "Loaded an unexpected dijet emulator module: "
            f"{loaded_file}"
        )
    return dijet_emulator


@lru_cache(maxsize=1)
def _dijet_cross_section_function() -> Callable[..., object]:
    """Return the legacy dijet-emulator public cross-section callable."""
    return _dijet_emulator_module().dijet_cross_section


@lru_cache(maxsize=1)
def inelastic_emulator_backend_id() -> str:
    """Return a checksum identity for cache invalidation."""
    digest = hashlib.sha256()
    for path in (DIJET_EMULATOR_SOURCE, DIJET_EMULATOR_CHECKPOINT):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return f"dijet_emulator:{digest.hexdigest()}"


def inelastic_emulator_device() -> str:
    """Return the PyTorch device selected by the legacy emulator."""
    return str(_dijet_emulator_module().device)


def _numeric_scalar_or_array(name: str, value: object) -> np.ndarray:
    """Convert one supported scalar or NumPy array to float64."""
    if not (np.isscalar(value) or isinstance(value, np.ndarray)):
        raise TypeError(
            f"{name} must be a scalar or a NumPy array; "
            f"received {type(value).__name__}."
        )
    try:
        # A private writable copy prevents read-only pandas/NumPy views from
        # triggering undefined-behavior warnings in the PyTorch emulator.
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if array.ndim > 0 and array.size == 0:
        raise ValueError(f"{name} must not be an empty array.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def quark_charge_squared(pdg: object) -> float | np.ndarray:
    """Return the squared fractional charge for a quark or antiquark PDG ID."""
    pdg_array = _numeric_scalar_or_array("pdg", pdg)
    absolute_pdg = np.abs(pdg_array)
    if np.any(absolute_pdg != np.floor(absolute_pdg)):
        raise ValueError("pdg must contain integer quark or antiquark PDG IDs.")

    absolute_pdg = absolute_pdg.astype(np.int64)
    if np.any(~np.isin(absolute_pdg, (1, 2, 3, 4, 5))):
        raise ValueError(
            "pdg must contain only quark or antiquark IDs: +/-1, ..., +/-5."
        )
    result = np.where(np.isin(absolute_pdg, (2, 4)), 4.0 / 9.0, 1.0 / 9.0)
    return float(result) if result.ndim == 0 else result


def _prepare_inputs(**values: object) -> tuple[Mapping[str, np.ndarray], tuple[int, ...] | None]:
    """Enforce equal array shapes and expand scalar inputs."""
    converted = {
        name: _numeric_scalar_or_array(name, value)
        for name, value in values.items()
    }
    array_shapes = [array.shape for array in converted.values() if array.ndim > 0]
    if array_shapes:
        target_shape = array_shapes[0]
        mismatches = [shape for shape in array_shapes[1:] if shape != target_shape]
        if mismatches:
            shape_description = ", ".join(
                f"{name}={array.shape}"
                for name, array in converted.items()
                if array.ndim > 0
            )
            raise ValueError(
                "All array inputs must have exactly the same shape; received "
                f"{shape_description}."
            )
        expanded = {
            name: (
                np.full(target_shape, float(array), dtype=np.float64)
                if array.ndim == 0
                else array
            )
            for name, array in converted.items()
        }
        return expanded, target_shape

    return {
        name: np.asarray(float(array), dtype=np.float64)
        for name, array in converted.items()
    }, None


def _validate_common(inputs: Mapping[str, np.ndarray]) -> None:
    """Validate the common DIS variables and polarization labels."""
    pol = inputs["pol"]
    if np.any((pol != 0.0) & (pol != 1.0)):
        raise ValueError("pol must contain only 0 (transverse) or 1 (longitudinal).")
    if np.any((inputs["y"] <= 0.0) | (inputs["y"] > 1.0)):
        raise ValueError("y must satisfy 0 < y <= 1.")
    if np.any(inputs["Q2"] <= 0.0):
        raise ValueError("Q2 must be strictly positive.")
    if np.any((inputs["z"] <= 0.0) | (inputs["z"] >= 1.0)):
        raise ValueError("z must satisfy 0 < z < 1.")
    if np.any(inputs["xg"] <= 0.0):
        raise ValueError("xg must be strictly positive.")


def _restore_output(values: np.ndarray, shape: tuple[int, ...] | None) -> float | np.ndarray:
    """Return a Python float for scalar calls or the original array shape."""
    values = np.asarray(values, dtype=np.float64)
    if shape is None:
        return float(values.reshape(-1)[0])
    return values.reshape(shape)


@lru_cache(maxsize=8)
def _radial_grid(
    n_radial: int = DEFAULT_N_RADIAL,
    ell_max: float = DEFAULT_ELL_MAX,
    ell_scale: float = DEFAULT_ELL_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the notebook's mapped Gauss--Legendre radial rule."""
    nodes, weights = leggauss(int(n_radial))
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    u_max = float(ell_max) / (float(ell_scale) + float(ell_max))
    u = u_max * unit_nodes
    ell = float(ell_scale) * u / (1.0 - u)
    d_ell_d_unit = float(ell_scale) * u_max / (1.0 - u) ** 2
    return ell, unit_weights * ell * d_ell_d_unit


def _angular_integrals(
    k: np.ndarray, ell: np.ndarray, eps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the analytic transverse and longitudinal angular integrals."""
    a = k**2 + ell**2 + eps**2
    delta_squared = np.maximum(
        a**2 - 4.0 * k**2 * ell**2,
        np.finfo(np.float64).tiny,
    )
    delta = np.sqrt(delta_squared)
    c = k**2 - ell**2 - eps**2

    angular_L = 2.0 * np.pi / delta
    with np.errstate(divide="ignore", invalid="ignore"):
        direct_T = np.pi * (1.0 + c / delta) / k
        stable_T = 4.0 * np.pi * k * eps**2 / (delta * (delta - c))
    angular_T = np.where(k == 0.0, 0.0, np.where(c < 0.0, stable_T, direct_T))
    return angular_T, angular_L


def _dipole_matrix(ell: np.ndarray, xg: np.ndarray) -> np.ndarray:
    """Evaluate one dipole row per xg value in bounded model batches."""
    shape = (xg.size, ell.size)
    flat_ell = np.broadcast_to(ell[None, :], shape).reshape(-1)
    flat_xg = np.broadcast_to(xg[:, None], shape).reshape(-1)
    output = np.empty(flat_ell.size, dtype=np.float64)
    for start in range(0, output.size, _DIPOLE_CALL_BATCH_SIZE):
        stop = min(start + _DIPOLE_CALL_BATCH_SIZE, output.size)
        output[start:stop] = np.asarray(
            dipole(flat_ell[start:stop], flat_xg[start:stop]),
            dtype=np.float64,
        )
    return output.reshape(shape)


def _elastic_photon_components(
    Q2: np.ndarray,
    z: np.ndarray,
    k1x: np.ndarray,
    k1y: np.ndarray,
    xg: np.ndarray,
    charge_squared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate both elastic photon components with the notebook algorithm."""
    Q2_flat = Q2.reshape(-1)
    z_flat = z.reshape(-1)
    k_flat = np.hypot(k1x.reshape(-1), k1y.reshape(-1))
    xg_flat = xg.reshape(-1)
    charge_flat = charge_squared.reshape(-1)
    eps_flat = np.sqrt(z_flat * (1.0 - z_flat) * Q2_flat)
    ell, radial_weights = _radial_grid()

    photon_T = np.empty(k_flat.size, dtype=np.float64)
    photon_L = np.empty(k_flat.size, dtype=np.float64)
    unique_xg, inverse_xg = np.unique(xg_flat, return_inverse=True)

    for xg_start in range(0, unique_xg.size, _DIPOLE_XG_BLOCK_SIZE):
        xg_stop = min(xg_start + _DIPOLE_XG_BLOCK_SIZE, unique_xg.size)
        xg_block = unique_xg[xg_start:xg_stop]
        dipole_block = _dipole_matrix(ell, xg_block)
        block_event_indices = np.flatnonzero(
            (inverse_xg >= xg_start) & (inverse_xg < xg_stop)
        )

        for event_start in range(
            0, block_event_indices.size, _ELASTIC_EVENT_BATCH_SIZE
        ):
            event_stop = min(
                event_start + _ELASTIC_EVENT_BATCH_SIZE,
                block_event_indices.size,
            )
            indices = block_event_indices[event_start:event_stop]
            k = k_flat[indices, None]
            eps = eps_flat[indices, None]
            event_dipole = dipole_block[inverse_xg[indices] - xg_start]
            angular_T, angular_L = _angular_integrals(
                k, ell[None, :], eps
            )
            weighted_dipole = radial_weights[None, :] * event_dipole
            convolution_T = np.sum(
                angular_T * weighted_dipole, axis=1
            ) / (2.0 * np.pi) ** 2
            convolution_L = np.sum(
                angular_L * weighted_dipole, axis=1
            ) / (2.0 * np.pi) ** 2

            bare_T = k_flat[indices] / (
                k_flat[indices] ** 2 + eps_flat[indices] ** 2
            )
            bare_L = 1.0 / (
                k_flat[indices] ** 2 + eps_flat[indices] ** 2
            )
            kernel_T = (bare_T - convolution_T) ** 2
            kernel_L = (bare_L - convolution_L) ** 2

            common = NC * ALPHA_EM * charge_flat[indices] * S_PERP
            photon_T[indices] = (
                2.0
                * common
                * (z_flat[indices] ** 2 + (1.0 - z_flat[indices]) ** 2)
                * kernel_T
                / (2.0 * np.pi) ** 2
            )
            photon_L[indices] = (
                8.0
                * common
                * z_flat[indices] ** 2
                * (1.0 - z_flat[indices]) ** 2
                * Q2_flat[indices]
                * kernel_L
                / (2.0 * np.pi) ** 2
            )

    return photon_T, photon_L


def el_cross_section(
    pol: object,
    y: object,
    Q2: object,
    z: object,
    k1x: object,
    k1y: object,
    xg: object,
    pdg: object,
) -> float | np.ndarray:
    """Return the selected elastic electron-level differential cross section.

    The returned measure is ``dy dQ2 dz d2k1``.  ``pol=0`` returns only the
    transverse term and ``pol=1`` returns only the longitudinal term.  ``pdg``
    supplies the quark or antiquark PDG ID used to determine the charge.
    """
    inputs, shape = _prepare_inputs(
        pol=pol,
        y=y,
        Q2=Q2,
        z=z,
        k1x=k1x,
        k1y=k1y,
        xg=xg,
        pdg=pdg,
    )
    _validate_common(inputs)
    charge_squared = np.asarray(
        quark_charge_squared(inputs["pdg"]), dtype=np.float64
    )
    if np.any(inputs["xg"] > X0_DIPOLE):
        raise ValueError(
            f"The elastic dipole requires xg <= {X0_DIPOLE}."
        )

    photon_T, photon_L = _elastic_photon_components(
        inputs["Q2"],
        inputs["z"],
        inputs["k1x"],
        inputs["k1y"],
        inputs["xg"],
        charge_squared,
    )
    y_flat = inputs["y"].reshape(-1)
    Q2_flat = inputs["Q2"].reshape(-1)
    flux = ALPHA_EM / (np.pi * Q2_flat * y_flat)
    result_T = flux * (1.0 - y_flat + 0.5 * y_flat**2) * photon_T
    result_L = flux * (1.0 - y_flat) * photon_L
    result = np.where(inputs["pol"].reshape(-1) == 0.0, result_T, result_L)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("The elastic cross section is non-finite.")
    return _restore_output(result, shape)



def _inelastic_emulator_components(
    Q2: np.ndarray,
    z: np.ndarray,
    k1x: np.ndarray,
    k1y: np.ndarray,
    k2x: np.ndarray,
    k2y: np.ndarray,
    xg: np.ndarray,
    charge_squared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the pointwise inelastic photon cross sections without q integration."""
    Q = np.sqrt(Q2.reshape(-1))
    z_flat = z.reshape(-1)
    xg_flat = xg.reshape(-1)
    charge_flat = charge_squared.reshape(-1)
    k1x_flat = k1x.reshape(-1)
    k1y_flat = k1y.reshape(-1)
    k2x_flat = k2x.reshape(-1)
    k2y_flat = k2y.reshape(-1)

    qx = k1x_flat + k2x_flat
    qy = k1y_flat + k2y_flat
    k1 = np.hypot(k1x_flat, k1y_flat)
    q = np.hypot(qx, qy)
    if np.any(k1 <= 0.0):
        raise ValueError("The inelastic emulator requires |k1| > 0.")
    if np.any(q <= 0.0):
        raise ValueError("The inelastic emulator requires |k1 + k2| > 0.")

    cosine = np.clip((k1x_flat * qx + k1y_flat * qy) / (k1 * q), -1.0, 1.0)
    theta = np.arccos(cosine)
    raw_T = np.empty(k1.size, dtype=np.float64)
    raw_L = np.empty(k1.size, dtype=np.float64)
    emulator = _dijet_cross_section_function()
    for start in range(0, k1.size, _EMULATOR_BATCH_SIZE):
        stop = min(start + _EMULATOR_BATCH_SIZE, k1.size)
        batch_T, batch_L = emulator(
            xg_flat[start:stop],
            Q[start:stop],
            z_flat[start:stop],
            k1[start:stop],
            q[start:stop],
            theta[start:stop],
            grid=False,
        )
        raw_T[start:stop] = np.asarray(batch_T, dtype=np.float64).reshape(-1)
        raw_L[start:stop] = np.asarray(batch_L, dtype=np.float64).reshape(-1)

    common = (
        2.0
        * NC
        * ALPHA_EM
        * charge_flat
        * S_PERP
        / (2.0 * np.pi) ** 6
    )
    return common * raw_T, common * raw_L


def inel_cross_section(
    pol: object,
    y: object,
    Q2: object,
    z: object,
    k1x: object,
    k1y: object,
    k2x: object,
    k2y: object,
    xg: object,
    pdg: object,
) -> float | np.ndarray:
    """Return the selected inelastic electron-level differential cross section.

    The returned measure is ``dy dQ2 dz d2k1 d2k2``.  The emulator is called
    directly at each Cartesian momentum point; no integration over q is done.
    ``pdg`` supplies the quark or antiquark PDG ID used to determine the charge.
    """
    inputs, shape = _prepare_inputs(
        pol=pol,
        y=y,
        Q2=Q2,
        z=z,
        k1x=k1x,
        k1y=k1y,
        k2x=k2x,
        k2y=k2y,
        xg=xg,
        pdg=pdg,
    )
    _validate_common(inputs)
    charge_squared = np.asarray(
        quark_charge_squared(inputs["pdg"]), dtype=np.float64
    )
    photon_T, photon_L = _inelastic_emulator_components(
        inputs["Q2"],
        inputs["z"],
        inputs["k1x"],
        inputs["k1y"],
        inputs["k2x"],
        inputs["k2y"],
        inputs["xg"],
        charge_squared,
    )

    y_flat = inputs["y"].reshape(-1)
    Q2_flat = inputs["Q2"].reshape(-1)
    flux = ALPHA_EM / (np.pi * Q2_flat * y_flat)
    result_T = flux * (1.0 - y_flat + 0.5 * y_flat**2) * photon_T
    result_L = flux * (1.0 - y_flat) * photon_L
    result = np.where(inputs["pol"].reshape(-1) == 0.0, result_T, result_L)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("The inelastic cross section is non-finite.")
    return _restore_output(result, shape)


elastic_cross_section = el_cross_section
inelastic_cross_section = inel_cross_section
