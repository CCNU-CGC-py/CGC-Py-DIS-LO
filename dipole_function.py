import copy
import torch
import numpy as np
import math
from NN_model import Net
from HankelIntegrator import HankelIntegratorPhysical 


def _parse_cuda_arch(arch):
    try:
        value = int(arch.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return divmod(value, 10)


def _cuda_arch_is_supported(device_capability, arch_list):
    if not arch_list:
        return True

    device_major, device_minor = device_capability
    for arch in arch_list:
        if not arch.startswith(("sm_", "compute_")):
            continue

        parsed = _parse_cuda_arch(arch)
        if parsed is None:
            continue

        arch_major, arch_minor = parsed
        if arch_major == device_major and arch_minor <= device_minor:
            return True

    return False


def _select_device():
    if not torch.cuda.is_available():
        print("Running on the CPU")
        return torch.device("cpu")

    cuda_device = torch.device("cuda:0")
    try:
        device_capability = torch.cuda.get_device_capability(cuda_device)
        arch_list = torch.cuda.get_arch_list()
        if not _cuda_arch_is_supported(device_capability, arch_list):
            arch_name = f"sm_{device_capability[0]}{device_capability[1]}"
            print(
                f"CUDA device architecture {arch_name} is not supported by "
                "this PyTorch build; running on the CPU"
            )
            return torch.device("cpu")

        probe = torch.ones(1, device=cuda_device)
        _ = probe * probe
        torch.cuda.synchronize(cuda_device)
    except RuntimeError as exc:
        print(f"CUDA is visible but could not be initialized ({exc}); running on the CPU")
        return torch.device("cpu")

    print("Running on the GPU")
    return cuda_device


device = _select_device()

def load_models(path_model="./"):  # model path
    model = Net(2,2).to(device)
    try:
        checkpoint = torch.load(f"{path_model}model_ciBK_best.pt", map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(f"{path_model}model_ciBK_best.pt", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


models_dict = load_models()

def MLP_ep_vm_ft(input_tensor):
    return models_dict(input_tensor)

def _to_1d_tensor(x, device=device):
    if isinstance(x, torch.Tensor):
        return x.detach().to(device, dtype=torch.float32).view(-1)
    return torch.as_tensor(x, dtype=torch.float32, device=device).view(-1)


# ==========================================
# Coordinate space (r-space) 
# ==========================================
def PINN_dipole_r(r, Y, grid=None):
    is_numpy_input = isinstance(r, np.ndarray) or isinstance(Y, np.ndarray)
    
    r_tensor = _to_1d_tensor(r)
    Y_tensor = _to_1d_tensor(Y)
    
    if grid is None:
        grid = (len(r_tensor) != len(Y_tensor))
        
    if grid:
        Y_grid, r_grid = torch.meshgrid(Y_tensor, r_tensor, indexing='ij')
        
        r_flat = r_grid.reshape(-1, 1)
        Y_flat = Y_grid.reshape(-1, 1)
        
        u = torch.log(1 / r_flat)
        input_tensor = torch.cat((u, Y_flat), dim=1) 
        
        N_r_Y, _, _, _ = MLP_ep_vm_ft(input_tensor)
        
        output = N_r_Y.view(len(Y_tensor), len(r_tensor))
    else:
        try:
            r_brd, Y_brd = torch.broadcast_tensors(r_tensor, Y_tensor)
        except RuntimeError as e:
            raise ValueError(
                f"[Shape Mismatch] In grid=False mode, the input shapes cannot be broadcasted!\n"
                f"Length of r: {len(r_tensor)}, Length of Y: {len(Y_tensor)}.\n"
                f"To generate a full 2D Cartesian grid for the physical phase space, please set grid=True."
            ) from e
            
        r_flat = r_brd.reshape(-1, 1)
        Y_flat = Y_brd.reshape(-1, 1)
        
        u = torch.log(1 / r_flat)
        input_tensor = torch.cat((u, Y_flat), dim=1) 
        
        N_r_Y, _, _, _ = MLP_ep_vm_ft(input_tensor)
        
        output = N_r_Y.squeeze(-1) 
        
    if is_numpy_input:
        return output.detach().cpu().numpy()
    return output


# ==========================================
# WW coordinate-space input via automatic differentiation
# ==========================================
def _color_factors(Nc=3.0):
    """Return (C_F, C_A) with C_F = (Nc^2 - 1)/(2 Nc), C_A = Nc."""
    Nc = float(Nc)
    c_f = (Nc ** 2 - 1.0) / (2.0 * Nc)
    c_a = Nc
    return c_f, c_a


def _second_argument(x, x0_for_Y):
    """Map Bjorken x to the model's second input Y = ln(x0/x) (or x itself)."""
    if x0_for_Y is None:
        return float(x)
    return float(np.log(x0_for_Y / float(x)))


_model_cache = {torch.float32: models_dict}


def _model_for_dtype(dtype):
    """Return the trained model in the requested dtype (double copy is cached).

    Using ``torch.float64`` greatly reduces the round-off amplification that
    corrupts the twice-differentiated WW kernel at small r, where S_F -> 1 and
    Gamma -> 0.
    """
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in _model_cache:
        model = copy.deepcopy(models_dict).to(dtype)
        model.eval()
        _model_cache[dtype] = model
    return _model_cache[dtype]


def ww_kernel_r(r, x, *, x0_for_Y=0.03, eps=1e-10, dtype=torch.float32, return_parts=False):
    r"""Compute the WW kernel K(r, x) = laplacian(Gamma) / Gamma via autograd.

    With Gamma(r, x) = -log S_F(r, x), S_F = 1 - N_F(r, x), the transverse
    Laplacian is evaluated exactly through PyTorch autograd:

        laplacian(Gamma) = d^2 Gamma / d r^2 + (1/r) d Gamma / d r.

    Because the network uses smooth (softplus/sigmoid) activations, the second
    derivative is analytic and free of the grid noise of finite differences.

    Pass ``dtype=torch.float64`` for a double-precision evaluation, which
    suppresses the small-r oscillations caused by float32 round-off.

    Returns K by default; with ``return_parts=True`` also returns
    (S_F, Gamma, laplacian). Output type matches the input (numpy or tensor).
    """
    is_numpy_input = isinstance(r, np.ndarray) or isinstance(x, np.ndarray)

    Y_val = _second_argument(x, x0_for_Y)
    net = _model_for_dtype(dtype)

    if isinstance(r, torch.Tensor):
        base = r.detach().to(device=device, dtype=dtype).view(-1)
    else:
        base = torch.as_tensor(r, dtype=dtype, device=device).view(-1)
    r_leaf = base.clone().requires_grad_(True)
    Y_flat = torch.full_like(r_leaf, Y_val)

    u = torch.log(1.0 / r_leaf).view(-1, 1)
    input_tensor = torch.cat((u, Y_flat.view(-1, 1)), dim=1)

    N_r_Y = net(input_tensor)[0]
    S_F = torch.clamp(1.0 - N_r_Y, min=eps)
    gamma = -torch.log(S_F)

    (dG_dr,) = torch.autograd.grad(gamma.sum(), r_leaf, create_graph=True)
    (d2G_dr2,) = torch.autograd.grad(dG_dr.sum(), r_leaf, create_graph=True)

    laplacian = d2G_dr2 + dG_dr / r_leaf
    K = laplacian / torch.clamp(gamma, min=eps)

    if is_numpy_input:
        to_np = lambda t: t.detach().cpu().numpy()
        if return_parts:
            return to_np(K), to_np(S_F), to_np(gamma), to_np(laplacian)
        return to_np(K)

    if return_parts:
        return K, S_F, gamma, laplacian
    return K


def ww_alpha_s_f_r_autograd(r, x, *, S_perp=1.0, Nc=3.0, x0_for_Y=0.03, eps=1e-10, dtype=torch.float32):
    r"""Autograd version of the coordinate-space WW input ``alpha_s * F_WW``.

        alpha_s F_WW = (C_F S_perp) / (2 pi^2) * K(r, x) * [1 - S_F^(Nc/C_F)],

    with the transverse Laplacian inside K(r, x) evaluated by automatic
    differentiation. ``S_perp`` defaults to 1. Pass ``dtype=torch.float64`` for
    a double-precision evaluation that suppresses small-r round-off noise.
    """
    K, S_F, _, _ = ww_kernel_r(
        r, x, x0_for_Y=x0_for_Y, eps=eps, dtype=dtype, return_parts=True
    )
    c_f, c_a = _color_factors(Nc)
    exponent = c_a / c_f  # Nc / C_F

    if isinstance(K, np.ndarray):
        result = (
            c_f * S_perp / (2.0 * np.pi ** 2)
            * K
            * (1.0 - S_F ** exponent)
        )
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        c_f * S_perp / (2.0 * math.pi ** 2)
        * K
        * (1.0 - S_F ** exponent)
    )


# ==========================================
# Saturation scale Q_s(Y)
# ==========================================
def _dipole_N_scalar(r, Y):
    """Fundamental dipole amplitude N_F(r, Y) as a Python float (scalar r, Y)."""
    val = PINN_dipole_r(float(r), float(Y), grid=False)
    if torch.is_tensor(val):
        return float(val.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(val).reshape(-1)[0])


def solve_Qs_vectorized(Y_values, initial_bounds=(1e-4, 50.0), method='brentq'):
    r"""Solve for the saturation scale Q_s(Y) from the fundamental dipole.

    Q_s is defined by N_F(r = sqrt(2)/Q_s, Y) = 1 - e^{-1/2}. For every Y the
    scalar equation N_F - target = 0 is bracketed and solved with SciPy's
    ``root_scalar``; the upper bracket is doubled up to a few times if the two
    endpoints do not straddle the root.

    Returns (Qs_solutions, convergence_info) as NumPy arrays. Non-converged
    entries are set to NaN.
    """
    from scipy.optimize import root_scalar

    Y_values = np.asarray(Y_values, dtype=float)
    target_value = 1.0 - np.exp(-0.5)
    Qs_solutions = np.zeros_like(Y_values)
    convergence_info = []

    def equation(Qs, Y):
        r = np.sqrt(2.0) / Qs
        return _dipole_N_scalar(r, Y) - target_value

    for i, Y in enumerate(Y_values):
        def current_eq(Qs):
            return equation(Qs, Y)

        bounds = list(initial_bounds)
        max_retries = 3
        converged = False

        for _ in range(max_retries):
            try:
                val_low = current_eq(bounds[0])
                val_high = current_eq(bounds[1])

                if np.isnan(val_low) or np.isnan(val_high):
                    raise ValueError("Neural network returned NaN at bracket boundaries.")

                if val_low * val_high > 0:
                    bounds[1] *= 2.0
                    continue

                result = root_scalar(current_eq, bracket=bounds, method=method)
                if result.converged:
                    Qs_solutions[i] = result.root
                    convergence_info.append(True)
                    converged = True
                    break
            except ValueError:
                pass

        if not converged:
            Qs_solutions[i] = np.nan
            convergence_info.append(False)
            print(f"Warning: Failed to converge when Y={Y:.4f}. Final bracket tried: {bounds}")

    return Qs_solutions, np.array(convergence_info)


def calc_Qs2(Y_array):
    """Return the squared saturation scale Q_s^2(Y)."""
    Qs_sols, _ = solve_Qs_vectorized(Y_array)
    return Qs_sols ** 2


QS2_TABLE_DEFAULT = "Qs2_table.dat"


def save_Qs2_table(x_array, Qs2, path=QS2_TABLE_DEFAULT):
    """Save (x, Q_s^2) columns to a .dat file for later interpolation.

    Non-finite rows (failed root solves) are dropped. Returns the path.
    """
    x_array = np.asarray(x_array, dtype=float).reshape(-1)
    Qs2 = np.asarray(Qs2, dtype=float).reshape(-1)
    mask = np.isfinite(x_array) & np.isfinite(Qs2)
    data = np.column_stack([x_array[mask], Qs2[mask]])
    header = "x    Qs2[GeV^2]   (saturation-scale table; interpolate Qs2 in log x)"
    np.savetxt(path, data, header=header, fmt="%.10e")
    return path


def load_Qs2_interpolator(path=QS2_TABLE_DEFAULT, kind="cubic"):
    """Read a saved (x, Q_s^2) table and return an interpolating callable.

    Q_s^2 is interpolated as a function of ``log(x)`` (the natural, smooth
    variable). The returned function ``Qs2_interp(x)`` accepts a scalar or an
    array of x and returns Q_s^2, extrapolating outside the tabulated range.
    """
    from scipy.interpolate import interp1d

    data = np.loadtxt(path)
    x = data[:, 0]
    qs2 = data[:, 1]
    order = np.argsort(x)
    log_x = np.log(x[order])
    qs2 = qs2[order]
    spline = interp1d(
        log_x, qs2, kind=kind, bounds_error=False, fill_value="extrapolate"
    )

    def Qs2_interp(x_query):
        xq = np.asarray(x_query, dtype=float)
        out = spline(np.log(xq))
        return float(out) if np.ndim(x_query) == 0 else out

    return Qs2_interp


# ==========================================
# Momentum space (k-space) 
# ==========================================
phys_integrator = HankelIntegratorPhysical(MLP_ep_vm_ft, N=2000, ht_fixed=0.0015, device=device)

def PINN_dipole_k(k_FT, Y_FT, representation='fundamental', grid=None):
    is_numpy_input = isinstance(Y_FT, np.ndarray) or isinstance(k_FT, np.ndarray)
    
    Y_tensor = _to_1d_tensor(Y_FT)
    k_tensor = _to_1d_tensor(k_FT)
    
    if grid is None:
        grid = (len(Y_tensor) != len(k_tensor))
        
    if grid:
        Y_grid, k_grid = torch.meshgrid(Y_tensor, k_tensor, indexing='ij')
        
        Y_flat = Y_grid.reshape(-1)
        k_flat = k_grid.reshape(-1)
        
        phi_F = phys_integrator(Y_flat, k_flat, representation=representation)
        phi_F = phi_F * (2 * math.pi)**2
        
        output = phi_F.view(len(Y_tensor), len(k_tensor))
    else:
        try:
            Y_brd, k_brd = torch.broadcast_tensors(Y_tensor, k_tensor)
        except RuntimeError as e:
            raise ValueError(
                f"[Shape Mismatch] In grid=False mode, the input shapes cannot be broadcasted!\n"
                f"Length of Y: {len(Y_tensor)}, Length of k: {len(k_tensor)}."
            ) from e
            
        phi_F = phys_integrator(Y_brd, k_brd, representation=representation)
        output = phi_F * (2 * np.pi)**2
        
    if is_numpy_input:
        return output.detach().cpu().numpy()
    return output
