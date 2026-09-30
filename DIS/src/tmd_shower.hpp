// ============================================================================
// tmd_shower.hpp
//
// Standalone PB-TMD initial-state backward parton shower.
//
// This is a self-contained C++ translation of the CASCADE 3.3.3 initial-state
// TMD shower (src/cascps.F + casbran_new.F + splitting kernels + alpha_s),
// following the Parton-Branching method of
//   Hautmann, Jung, Lelek, Radescu, Zlebcik, arXiv:1708.03279 (DESY 17-118).
//
// PHYSICS ROLE
//   Given an incoming parton that enters a hard process with longitudinal
//   fraction x and transverse momentum k_T (for CGC: the Wilson-line-correlator
//   sampled gluon), this backward-evolves it from the hard scale mu_hard DOWN
//   to an infrared/saturation scale Q0, accumulating the transverse recoil of
//   the emitted partons:
//                       Q  =  - sum_c q_c          (Eq. 36 of the paper)
//   "Q" is the extra initial-state-radiation transverse momentum that the
//   mu-evolution between Q0 and mu_hard adds on top of the input density.
//
//   Near back-to-back the Sudakov form factor self-suppresses large recoil;
//   away from back-to-back it does little. There is NO manual regime switch.
//
// DESIGN
//   * Process-agnostic: no CGC / Dihadron / DIS / LHAPDF / Pythia / GSL types.
//   * The parton density is injected as a std::function callback, so the same
//     module serves the dihadron generator, the DIS dijet generator, or any
//     future process. For CGC, wrap your Wilson-line correlator interpolation.
//   * Scope: gluon line only, LO g->gg kernel (by design — no flavor-changing
//     channels and no NLO kernels). The emitted soft partons are KEPT with
//     full lab-frame 4-momenta and returned in hard-to-beam order so the
//     caller can hadronize them as the target-side color string. The net
//     recoil Q = -sum q_c is still provided as well.
//
// DOCUMENTED APPROXIMATIONS (see also KNOWN_ISSUES.md style)
//   A1. The CGC correlator is treated as the saturation-scale INPUT density,
//       even though it already contains its own high-k_T tail. The shower then
//       supplies the mu-evolution from Q0 up to mu_hard. This is the agreed
//       "input-scale" choice; it is an approximation, not exact matching.
//   A2. The TMD density enters through the backward acceptance weight as a ratio
//       f(x',k'^2,mu) / f(x,k^2,mu) (CASCADE line 451). In CASCADE's multi-flavor
//       implementation, the ratio f(i)/f(iflb) appears in both the scale generation
//       (line 276) and acceptance weight (line 451), where it CANCELS. The net effect:
//       TMD does NOT modulate the Sudakov evolution rate; it only affects the
//       acceptance probability through the density ratio f'/f. This is the standard
//       parton shower prescription where the density ratio filters proposed branchings.
//   A3. Emitted parton longitudinal kinematics use the standard backward-shower
//       light-cone reconstruction: in branching b->a+c the emitted c carries
//       (1-z) of the parent's beam light-cone momentum, transverse q_c, put
//       on-shell (massless gluon). Recoil is local to each branching (no
//       global momentum reshuffling). Adequate for soft target-side radiation.
//
// USAGE (any process)
//   #include "tmd_shower.hpp"
//   using namespace tmdshower;
//
//   Config cfg;                       // tune rootS, Lambda, Q0, ...
//   cfg.rootS = 200.0;
//   // density: id, x, kt2, mu  ->  density value (mu may be ignored, see A2)
//   DensityFn dens = [](int id, double x, double kt2, double mu) {
//       (void)mu;                     // CGC correlator has no mu
//       return cgc_target_density(x, std::sqrt(kt2));   // your wrapper
//   };
//   TMDShower shower(cfg, dens, /*seed=*/12345);
//
//   Result r = shower.run(/*idIn=*/21, xB, ktx, kty, /*muHard=*/PT);
//   if (r.ok) { /* recoil the outgoing hard system by (r.Qx, r.Qy) */ }
//
// COMPILE
//   g++ -O3 -std=c++17 -c tmd_shower.cc        # no extra include/lib needed
//   ... link tmd_shower.o into the generator that calls it.
// ============================================================================
#ifndef TMD_SHOWER_HPP
#define TMD_SHOWER_HPP

#include <functional>
#include <vector>
#include <random>
#include "Pythia8/StandardModel.h"   // Pythia8::AlphaStrong

namespace tmdshower {

// ----------------------------------------------------------------------------
// Density callback.
//   id  : PDG id of the parton whose density is requested (21 = gluon)
//   x   : longitudinal momentum fraction
//   kt2 : transverse momentum squared [GeV^2]
//   mu  : evolution scale [GeV]   (may be ignored; see approximation A2)
// Returns the (T)MD density value at that point. Only ratios are used, so the
// normalization is irrelevant; it must be positive where physical.
// ----------------------------------------------------------------------------
using DensityFn = std::function<double(int id, double x, double kt2, double mu)>;

// ----------------------------------------------------------------------------
// Shower configuration. Defaults are sensible for RHIC sqrt(s)=200 GeV CGC.
// ----------------------------------------------------------------------------
struct Config {
    double rootS    = 200.0;     // sqrt(s) [GeV]
    double alphaS_MZ    = 0.118; // alpha_s(M_Z) for Pythia8::AlphaStrong
    int    alphaS_order = 1;     // 0=fixed, 1=LO, 2=NLO, 3=NNLO (Pythia)
    double Q0       = 0.5;       // shower IR cutoff [GeV]; set per event via
                                 //   TMDShower::setQ0() from the caller, using
                                 //   Qs(x_hard) so the shower start tracks the
                                 //   saturation scale of the current event.
    double xMax     = 0.97;      // stop backward evolution when x >= xMax
    double zMax     = 0.99999;   // soft-gluon resolution z_M upper bound
    double Bfac     = 10.0;      // veto-algorithm enhancement factor (CASCADE)
    int    maxBranch = 1000;     // safety cap: max branchings per leg
    int    maxTry    = 200000;   // safety cap: max veto tries per branching
    bool   enabled   = true;     // master switch; if false run() returns Q=0
};

// One recorded branching b -> a + c (the emitted parton is c).
// (qx,qy) are also the (px,py) of the emitted parton; pz,E complete its
// lab-frame 4-momentum (massless gluon, see approximation A3).
struct Emission {
    double qx = 0.0, qy = 0.0;   // emitted parton transverse momentum [GeV]
    double pz = 0.0, E  = 0.0;   // emitted parton longitudinal mom. & energy
    double z  = 0.0;             // splitting fraction
    double mu = 0.0;             // scale of this branching [GeV]
    int    idEmit = 21;          // PDG id of emitted parton (always 21 = gluon)
};

// Result of a backward evolution.
struct Result {
    bool   ok   = false;         // true if evolution completed normally
    double Qx   = 0.0;           // net recoil = -sum q_c, x-component [GeV]
    double Qy   = 0.0;           // net recoil, y-component [GeV]
    double xEnd = 0.0;           // x after backward evolution
    int    idEnd = 21;           // flavor after backward evolution (always 21)
    int    nBranch = 0;          // number of branchings generated
    // Emitted partons in evolution order = HARD-TO-BEAM order along the
    // ladder: chain.front() is the highest-scale branching (nearest the hard
    // vertex), chain.back() the lowest-scale (nearest the beam). This is the
    // order to color-connect them as a gluon ladder; note scale order does
    // NOT track emitted-parton energy.
    std::vector<Emission> chain;
};

// ----------------------------------------------------------------------------
// The shower.
// ----------------------------------------------------------------------------
class TMDShower {
public:
    TMDShower(const Config& cfg, DensityFn density, unsigned seed = 0);

    // Backward-evolve one incoming parton.
    //   idIn     : PDG id entering the hard process (must be 21)
    //   x        : its longitudinal momentum fraction
    //   ktx,kty  : its transverse momentum [GeV] (CGC: sampled imbalance)
    //   muHard   : scale to start the backward evolution from [GeV] (e.g. P_T)
    //   beamSign : +1 if the showered leg moves along +z (projectile),
    //              -1 if along -z (CGC target). Sets emitted-parton pz sign.
    Result run(int idIn, double x, double ktx, double kty, double muHard,
               int beamSign = -1);

    // Override the per-event IR cutoff before calling run(). Intended use:
    // caller computes Qs(x_hard) and calls setQ0(Qs) so the shower terminates
    // at the saturation scale of the current event. Frozen for the whole chain.
    void setQ0(double q0) { cfg_.Q0 = q0; }

    // Override the per-event hard-system invariant energy.  DIS samples W2
    // event by event, so the shower positivity veto must use sqrt(W2) for the
    // current event rather than a run-wide fixed value.
    void setRootS(double rootS) { cfg_.rootS = rootS; }

private:
    Config      cfg_;
    DensityFn   density_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uni_{0.0, 1.0};
    mutable Pythia8::AlphaStrong asPy_;  // running coupling from Pythia

    double rand01() { return uni_(rng_); }

    // alpha_s(mu^2) via Pythia8::AlphaStrong (initialized in the ctor).
    double alphaS(double mu2) const;

    // LO splitting function P_{ab}(z). Gluon-only by design: only g->gg
    // returns nonzero; all other (idMother, idDaughter) combinations -> 0.
    double splittingP(int idMother, int idDaughter, double z) const;

    // Generate one backward branching using the CASCADE veto algorithm.
    // Updates (id, x, ktx, kty, mu) in place and fills `em`.
    // Returns false when no further resolvable branching is found.
    bool oneBranching(int& id, double& x, double& ktx, double& kty,
                       double& mu, int beamSign, Emission& em);
};

// ----------------------------------------------------------------------------
// Recoil application: exact Lorentz transformation of the hard pair via
// Pythia's RotBstMatrix.
//
// Callers run `gShower->run(...)` and absorb the returned transverse recoil
// (sh.Qx, sh.Qy) into the hard pair (p1, p2). This helper performs the exact
// Lorentz map that shifts P = p1 + p2 -> P_new, where P_new has the same
// invariant mass M and transverse momentum increased by (Qx, Qy).
//
// Implementation builds the transformation as a Pythia RotBstMatrix:
//   M.bstback(P)  : composes a boost from lab -> rest frame of P
//   M.bst(P_new)  : composes a boost from rest frame of P_new -> lab
// Then `rotbst(M)` applies the same composed transformation to each parton.
// Each parton's mass is preserved; the pair's invariant mass is preserved.
// ----------------------------------------------------------------------------
inline void applyRecoil(Pythia8::Vec4& p1, Pythia8::Vec4& p2,
                        double Qx, double Qy) {
    if (Qx == 0.0 && Qy == 0.0) return;
    Pythia8::Vec4 P = p1 + p2;
    const double M2 = std::max(P.m2Calc(), 0.0);
    const double new_px = P.px() + Qx;
    const double new_py = P.py() + Qy;
    const double new_pz = P.pz();
    const double new_E  = std::sqrt(M2 + new_px*new_px + new_py*new_py + new_pz*new_pz);
    Pythia8::Vec4 P_new(new_px, new_py, new_pz, new_E);
    Pythia8::RotBstMatrix M;
    M.bstback(P);
    M.bst(P_new);
    p1.rotbst(M);
    p2.rotbst(M);
}

} // namespace tmdshower

#endif // TMD_SHOWER_HPP
