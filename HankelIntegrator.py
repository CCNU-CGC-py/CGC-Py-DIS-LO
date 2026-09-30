import torch
import torch.nn as nn
import numpy as np
from scipy.special import jv, yv, jn_zeros
from torch.utils.checkpoint import checkpoint

def precompute_j1(N: int = 1200, ht_fixed: float = 0.015):
    """
    [Derivative Method Core]
    Bessel-zeros based DE-mapped quadrature weights for J1.
    Used for the formula: int_0^inf x * J1(x) * f(x) dx
    ALWAYS use float64 for precomputation.
    """
    dtype_calc = np.float64
    
    # 使用 J1 的零点 (nu = 1.0)
    nu = 1.0 
    zeros = jn_zeros(nu, N).astype(dtype_calc)
    xi = (zeros / np.pi).astype(dtype_calc)
    
    # Ogata 权重公式对于 J1: w = Y1(pi*xi) / J2(pi*xi)
    # 注意 scipy 的 index: jv(nu+1) -> J2
    w = (yv(nu, zeros) / jv(nu + 1.0, zeros)).astype(dtype_calc)

    t = ht_fixed * xi
    
    # Double Exponential (DE) mapping
    psi = t * np.tanh((np.pi / 2.0) * np.sinh(t))
    knots = (np.pi / ht_fixed) * psi 

    # Jacobian d(knots)/dt
    a = (np.pi / 2.0) * np.sinh(t)
    cosh_t = np.cosh(t)
    tanh_a = np.tanh(a)
    psip = np.pi * t * (1.0 - tanh_a ** 2) * cosh_t / 2.0 + tanh_a
    psip[np.isnan(psip)] = 1.0 

    J1_val = jv(1.0, knots).astype(dtype_calc)
    
    weight = (np.pi * w * J1_val * psip).astype(dtype_calc)
    
    return knots, weight


class HankelIntegratorPhysical(nn.Module):
    """
    Compute phi(k, Y) using the Derivative Method (Integration by Parts).
    
    Fundamental:
      phi_F(k) = 1/(2π k^2) * ∫_0^∞ x J_1(x) * [ (1/k) * (dN/dr) ] dx
      
    Adjoint (S_A = S_F^(9/4)):
      phi_A(k) = 1/(2π k^2) * ∫_0^∞ x J_1(x) * [ -(1/k) * d(S_A)/dr ] dx
               = 1/(2π k^2) * ∫_0^∞ x J_1(x) * [ (1/k) * (9/4) * (1-N)^(5/4) * (dN/dr) ] dx
               
    Key Feature:
      dipole density,
      which vanishes at r->0 and r->inf.
    """
    def __init__(self, f_net, N: int = 1200, ht_fixed: float = 0.015, device=None, chunk_size: int = 512):
        super().__init__()
        self.f_net = f_net
        self.device = device
        self.chunk_size = int(chunk_size)

        knots_np, weight_np = precompute_j1(N=N, ht_fixed=ht_fixed)
        
        with np.errstate(divide='ignore'):
            log_knots_np = np.log(knots_np)
        log_knots_np[np.isneginf(log_knots_np)] = -1000.0 

        knots = torch.tensor(knots_np, dtype=torch.float32, device=self.device)
        weight = torch.tensor(weight_np, dtype=torch.float32, device=self.device)
        neg_log_knots = torch.tensor(-log_knots_np, dtype=torch.float32, device=self.device)

        self.register_buffer("knots", knots)
        self.register_buffer("weight", weight)
        self.register_buffer("neg_log_knots", neg_log_knots)
        self.register_buffer("kweight", knots * weight)

    @staticmethod
    def _unwrap_net_out(out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 2 and out.shape[1] == 1:
            out = out[:, 0]
        return out

    def forward(
        self,
        y_vec: torch.Tensor,
        k_vec: torch.Tensor,
        representation: str = "both",
        use_checkpoint=True,
        fp64_accum: bool = False,
    ):
        if y_vec.shape[0] != k_vec.shape[0]:
            if y_vec.shape[0] == 1:
                y_vec = y_vec.expand_as(k_vec)
            else:
                raise ValueError(f"Shape mismatch: y_vec {tuple(y_vec.shape)} vs k_vec {tuple(k_vec.shape)}")

        B = k_vec.shape[0]
        Nk = self.knots.shape[0]

        if use_checkpoint is None:
            use_checkpoint = bool(self.training)

        rep = representation.lower().strip()
        
        partial_F = []
        partial_A = [] 

        if fp64_accum:
            kweight = self.kweight.to(dtype=torch.float64)
        else:
            kweight = self.kweight
            
        Nc_over_Cf = 9.0 / 4.0  # 2.25

        for i in range(0, B, self.chunk_size):
            y_chunk_vec = y_vec[i : i + self.chunk_size]
            k_chunk_vec = k_vec[i : i + self.chunk_size]
            curr_b = y_chunk_vec.shape[0]

            logk = torch.log(torch.clamp(k_chunk_vec, min=1e-30)).view(curr_b, 1)
            u_input = self.neg_log_knots.view(1, Nk) + logk
            u_input = u_input.detach().requires_grad_(True)

            y_chunk = y_chunk_vec.view(curr_b, 1).expand(curr_b, Nk)
            inp = torch.stack([u_input, y_chunk], dim=-1).reshape(-1, 2)

            def custom_forward(input_data):
                return self._unwrap_net_out(self.f_net(input_data))

            if use_checkpoint:
                N_F_flat = checkpoint(custom_forward, inp, use_reentrant=False)
            else:
                N_F_flat = custom_forward(inp)
            
            grad_outputs = torch.ones_like(N_F_flat)
            grads = torch.autograd.grad(
                outputs=N_F_flat,
                inputs=inp,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            dN_du = grads[:, 0].reshape(curr_b, Nk)
            
            knots_expand = self.knots.view(1, Nk).expand(curr_b, Nk)
            k_expand = k_chunk_vec.view(curr_b, 1).expand(curr_b, Nk)
            r_val = knots_expand / k_expand
             
            dN_dr = dN_du * (-1.0 / r_val)
            
            inv_k = 1.0 / k_chunk_vec.view(curr_b, 1)
            
            # --- Fundamental ---
            # Integrand_F = (1/k) * dN/dr
            integrand_F = dN_dr * inv_k
            
            if fp64_accum:
                integrand_F_use = integrand_F.to(dtype=torch.float64)
            else:
                integrand_F_use = integrand_F

            chunk_int_F = torch.sum(integrand_F_use * kweight.view(1, Nk), dim=1)
            partial_F.append(chunk_int_F)
            
            # --- Adjoint ---
            if rep in {"adjoint", "both"}:     
                N_F_chunk = N_F_flat.reshape(curr_b, Nk)  
                S_F_chunk = torch.clamp(1.0 - N_F_chunk, min=1e-30) 
                
                factor_A = Nc_over_Cf * torch.pow(S_F_chunk, Nc_over_Cf - 1.0) # 2.25 * S^1.25     
                integrand_A = factor_A * integrand_F 
                
                if fp64_accum:
                    integrand_A_use = integrand_A.to(dtype=torch.float64)
                else:
                    integrand_A_use = integrand_A

                chunk_int_A = torch.sum(integrand_A_use * kweight.view(1, Nk), dim=1)
                partial_A.append(chunk_int_A)

        denom = (2.0 * np.pi) * (k_vec ** 2)
        
        # Fundamental Output
        integral_F = torch.cat(partial_F, dim=0)
        phi_F = integral_F / denom
        if fp64_accum and phi_F.dtype != k_vec.dtype:
            phi_F = phi_F.to(dtype=k_vec.dtype)
            
        if rep == "fundamental":
            return phi_F

        # Adjoint Output
        integral_A = torch.cat(partial_A, dim=0)
        phi_A = integral_A / denom
        if fp64_accum and phi_A.dtype != k_vec.dtype:
            phi_A = phi_A.to(dtype=k_vec.dtype)

        if rep == "adjoint":
            return phi_A

        return phi_F, phi_A