# CGC-py

**A Monte Carlo Event Generator for Gluon Saturation Physics in Deep-Inelastic Scattering**

[![arXiv](https://img.shields.io/badge/arXiv-2609.03434-b31b1b.svg)](https://arxiv.org/abs/2609.03434)

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
- For `x_g > x0 = 0.01`, the cross section and WW kernel are continued with
  `[(1 − x)/(1 − x0)]⁴`.

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
- **Pythia 8 runs with default (Monash 2013) parameters and no retuning.**

---

## Requirements

TODO: confirm versions.

- Python ≥ `<version>`
- [Pythia 8.3](https://pythia.org) with Python bindings
- NumPy, SciPy
- `<ML framework used for the emulator, e.g. PyTorch>`
- Dipole-amplitude tables (proton, Au) from the PINN fit
- Pretrained inelastic cross-section emulator weights

## Installation

```bash
git clone <repo-url>
cd CGC-py
pip install -r requirements.txt        # TODO
# TODO: instructions for obtaining dipole tables and emulator weights
```

---

## Quick start

TODO: replace with the real entry point and options.

```bash
python <run_script>.py --config configs/<example>.yaml --nevents 100000 --output events.<ext>
```

Example configuration (illustrative keys only):

```yaml
beams:
  Ee: 27.6          # electron energy [GeV]
  EN: 920.0         # energy per nucleon [GeV]
  target: p         # p | Au
kinematics:
  Q2:  [5.0, 10.0]  # GeV^2
  y:   [0.05, 0.6]
frame: cm           # cm (γ*-target CM) | rest (target rest frame)
shower:
  Q0: 2.5           # PB-TMD cutoff [GeV]; value used in the paper
  x0: 0.01          # large-x continuation boundary
pythia:
  tune: default     # Monash 2013, no retuning
```

### Key parameters

| Parameter | Meaning | Paper value |
|-----------|---------|-------------|
| `Q0` | PB–TMD evolution cutoff and resolvability scale | 2.5 GeV |
| `x0` | Tabulation boundary for WW kernel continuation | 0.01 |
| `S⊥^Au / (A S⊥^p)` | Geometric prefactor for `R_eAu` | 1/2.57 |
| Quark flavors | Light flavors, massless | u, d, s |

---

## Output

Each event contains the scattered-electron kinematics (`y`, `Q²`, `x_B`, `x_g`), the
sampled q q̄ configuration with its CGC weight, the PB–TMD recoil, and all stable hadrons
after Pythia 8. TODO: document the file format and fields.

---

## Validation and results reproduced in the paper

| Study | Setup | Paper figure |
|-------|-------|--------------|
| Closure test: single-inclusive quark spectrum vs. analytic result | ep, eAu; 27.6 × 920 GeV; 0.05<y<0.6; 5<Q²<10 GeV² | Fig. 3 |
| `x_B` vs. `x_g` distributions | Ee = 20 GeV; Ep = 250 / 1000 / 25000 GeV | Fig. 4 |
| Charged-hadron `p*_T` spectra vs. H1 data | 27.6 × 920 GeV; 1.5<η*<5 | Fig. 5 |
| Nuclear modification factor `R^h_eAu` | Same bins as Fig. 5 | Fig. 6 |
| Dihadron `C_P(Δφ)` vs. Pythia 6, with `x_g` rescaling | 4<Q²<8 GeV², 0.6<y<0.8, 0.1<z_h<0.5, p_T^trig>2, 1<p_T^assoc<2 GeV | Fig. 7 |
| Dihadron `C_P(Δφ)`: ep vs. eAu | Same as Fig. 7 | Fig. 8 |

TODO: add scripts or commands to reproduce each figure.

---

## Current limitations

- Leading-order hard process (`γ* → q q̄` only). No `γ* g → q q̄ g` real emission or
  virtual corrections.
- No target remnant or backward-hemisphere hadron production. Use for current-region
  (forward) observables.
- Impact-parameter-independent target.
- Light massless quarks only. No heavy flavor yet.

Planned extensions include the target remnant, NLO hard processes, and dijet,
photon–hadron, energy-correlator, and heavy-flavor observables.

---

## Citation

If you use CGC-py, please cite:

```bibtex
@article{Duan:2026cgcpy,
  author        = {Duan, Haowu and Yi, Cong and Dai, Si-Wei and Wei, Shu-Yi and Zhao, Wenbin and Zheng, Liang},
  title         = {{CGC-py: A Monte Carlo Event Generator for Gluon Saturation Physics}},
  eprint        = {2609.03434},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ph},
  year          = {2026}
}
```

Please also cite the components CGC-py depends on: Pythia 8.3 (arXiv:2203.11601), the
PB–TMD method and CASCADE3 (arXiv:1708.03279, arXiv:2101.10221), the Filon-quadrature
correlator paper (arXiv:2608.18589), and the PINN dipole-amplitude fits
(arXiv:2603.08008, arXiv:2607.27603).

## License

TODO

## Contact

Maintainer emails：haowu.duan@ccnu.edu.cn; congyi@ccnu.edu.cn; swdai@mails.ccnu.edu.cn; wenbinzhao@ccnu.edu.cn
Affiliations: Central China Normal University; 