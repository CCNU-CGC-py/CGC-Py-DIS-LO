# CGC-py

**A Monte Carlo Event Generator for Gluon Saturation Physics in Deep-Inelastic Scattering**

[![arXiv](https://img.shields.io/badge/arXiv-2609.03434-b31b1b.svg)](https://arxiv.org/pdf/2609.03434)

CGC-py generates fully exclusive hadronic final states for deep-inelastic scattering
off protons and nuclei, `e + p(A) → e + q q̄ + X`, at small x. It couples:

1. **The full leading-order Color Glass Condensate (CGC) cross section** for
   `γ* p(A) → q q̄ + X`, including both target-elastic and target-inelastic contributions,
   for transversely and longitudinally polarized photons. It does **not** take the
   back-to-back (correlation) limit.
2. **A Parton-Branching TMD (PB–TMD) backward initial-state shower** from the small-x
   target, built on a scale-dependent Weizsäcker–Williams gluon distribution, with
   invariant-mass-preserving recoil of the q q̄ pair.
3. **Pythia 8** for final-state radiation and Lund string fragmentation.

Because the hard process is evaluated in full kinematics, **single-inclusive and dihadron
(or dijet) observables come from the same event sample with the same normalization.**
There is no need for separate calculations.

---

## Physics overview

```
 e⁻ ──γ*(Q²)──►  [ Full CGC q q̄ cross section ]  ──q, q̄──►  [ Pythia 8: FSR + Lund ]  ──► hadrons
                  (T/L; elastic + inelastic)                                               (π±, K±, p/p̄, jets…)
                              ▲
 target p/A ──small-x field──►[ PB–TMD backward target shower ] ── recoil Q_T on q q̄
```

### Hard process
- Dipole `S(r)` and quadrupole `S⁽⁴⁾` correlators. The quadrupole uses the large-Nc
  Gaussian approximation, split into an elastic part `S(r)S(r')` and an inelastic part.
- Impact-parameter-independent target with effective transverse area `S⊥`.
- The inelastic contribution is evaluated with an **emulator** trained on data generated
  by a Filon-type quadrature algorithm (arXiv:2608.18589).
- Dipole amplitudes for the proton and Au are taken from the **PINN global fit with
  collinear-improved BK evolution** (arXiv:2603.08008, arXiv:2607.27603).
- Light flavors `u, d, s` are treated as massless.
- Photon flux uses the standard `f_T(y,Q²)` and `f_L(y,Q²)` factors.

### Initial-state PB–TMD shower
- Only `g → gg` emissions, with the full `P_gg` kernel, one-loop running αs, and a
  TMD-density ratio in the Sudakov factor.
- Angular ordering: `μ' = |q_c|/(1 − z)`, `k' = k + q_c`.
- The shower starts at `μ_max = max(P⊥, Q0)` and evolves backward to the cutoff `Q0`.
- Veto algorithm with the eikonal overestimate `P_eik = 2C_A[1/z + 1/(1−z)]`.
- The accumulated recoil `Q_T = −Σ q_c,i` is applied to the q q̄ pair by a Lorentz
  transformation that preserves `M_qq̄` and on-shell conditions.
- Emitted target-side gluons are **not** tracked. They populate the backward hemisphere
  only.

### Final state
- Each hard parton is paired with a zero-momentum companion parton to form an
  independent color-singlet string.
- Time-like FSR starts at `P⊥` and is followed by Lund fragmentation.
- **Pythia 8 runs with default parameters and no retuning.**


## Citation

If you use CGC-py, please cite:

```bibtex
@article{Duan:2026jrk,
    author = "Duan, Haowu and Yi, Cong and Dai, Si-Wei and Wei, Shu-Yi and Zhao, Wenbin and Zheng, Liang",
    title = "{CGC-py: A Monte Carlo Event Generator for Gluon Saturation Physics}",
    eprint = "2609.03434",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "9",
    year = "2026"
}
```

Please also cite the components CGC-py depends on: Pythia 8.3 (arXiv:2203.11601), the
PB–TMD method and CASCADE3 (arXiv:1708.03279, arXiv:2101.10221), the Filon-quadrature
correlator paper (arXiv:2608.18589), and the PINN dipole-amplitude fits
(arXiv:2603.08008, arXiv:2607.27603).


## Contact

Maintainer emails：
Haowu Duan: haowu.duan@ccnu.edu.cn; 
Cong Yi: congyi@ccnu.edu.cn; 
Si-Wei Tai: swdai@mails.ccnu.edu.cn; 
Wenbin Zhao: wenbinzhao@ccnu.edu.cn
Affiliations: Central China Normal University; 
