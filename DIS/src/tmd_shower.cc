// ============================================================================
// tmd_shower.cc   --   implementation of the standalone PB-TMD shower
//
// Algorithm faithfully follows CASCADE 3.3.3:
//   cascps.F        : top-level backward-evolution loop      -> TMDShower::run
//   casbran_new.F   : single branching via the veto algorithm -> oneBranching
//   splitting kernel : LO P_gg (gluon-only by design)          -> splittingP
//   alphas.F        : one-loop running coupling                -> alphaS
//
// Equation numbers refer to arXiv:1708.03279 (the Parton-Branching paper):
//   Eq. 34  : next scale from the Sudakov form factor (here: veto algorithm)
//   Eq. 35  : splitting variable z from the splitting function
//   Eq. 36  : k = -sum_c q_c        (accumulated transverse recoil)
//   Eq. 41  : angular ordering  mu = |q_c| / (1-z)
// ============================================================================
#include "tmd_shower.hpp"

#include <cmath>
#include <algorithm>

namespace tmdshower {

namespace {
    constexpr double kCA   = 3.0;                 // SU(3) adjoint Casimir
    constexpr double kPi   = 3.14159265358979323846;
    constexpr double kTwoPi = 2.0 * kPi;
}

// ----------------------------------------------------------------------------
TMDShower::TMDShower(const Config& cfg, DensityFn density, unsigned seed)
    : cfg_(cfg), density_(std::move(density))
{
    if (seed == 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(seed);
    }
    // Pythia's running coupling: same engine as the FSR shower uses.
    asPy_.init(cfg_.alphaS_MZ, cfg_.alphaS_order);
}

// ----------------------------------------------------------------------------
// alpha_s(mu^2) via Pythia8::AlphaStrong (same coupling Pythia's showers use).
// Frozen at Q0^2 below Q0 to keep the veto-algorithm overestimate finite.
// ----------------------------------------------------------------------------
double TMDShower::alphaS(double mu2) const
{
    const double Q0eff2 = cfg_.Q0 * cfg_.Q0;
    return asPy_.alphaS(std::max(mu2, Q0eff2));
}
// ----------------------------------------------------------------------------
// LO splitting function. Gluon-only by design: only g -> g g implemented.
//   P_gg(z) = 2 C_A [ z/(1-z) + (1-z)/z + z(1-z) ]
// Flavor-changing channels (q->qg, g->qqbar) are intentionally out of scope.
// ----------------------------------------------------------------------------
double TMDShower::splittingP(int idMother, int idDaughter, double z) const
{
    if (idMother == 21 && idDaughter == 21) {
        return 2.0 * kCA * ( z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z) );
    }
    return 0.0;   // gluon-only by design; no quark channels
}

// ----------------------------------------------------------------------------
// One backward branching, CASCADE veto algorithm (casbran_new.F).
//
// Work in the evolution variable  t = ln(mu^2 / Lambda^2).  Backward evolution
// decreases t from the hard scale toward the cutoff.
//
// Overestimate used for the veto (so the scale step is analytically
// invertible):
//   P_gg(z) <= Phat(z) = 2 C_A [ 1/(1-z) + 1/z ]
//   alpha_s(mu^2) <= alpha_max = alpha_s(Q0^2)
// The true weight P/Phat * alpha/alpha_max * density-ratio (Eqs. 34-36) is
// applied as the accept probability; rejected trials continue the veto.
//
// Design simplifications relative to a faithful PB-TMD (arXiv:1708.03279):
//   - Gluon-only by design: only g -> g g; no flavor-changing channels.
//   - LO splitting function only; no NLO corrections, no NLL Sudakov.
//   - Input "density" is a CGC Wilson-line correlator, not a true PB-TMD;
//     the density-ratio in the weight is therefore only approximate.
//   - This file is duplicated between pA_gen/tmd_shower.cc and
//     DIS_gen/src/tmd_shower.cc. Both copies must stay byte-identical;
//     keep them in sync until a shared top-level common/ home is set up.
// ----------------------------------------------------------------------------
bool TMDShower::oneBranching(int& id, double& x, double& ktx, double& kty,
                              double& mu, int beamSign, Emission& em)
{
    if (id != 21) return false;                         // gluon-only by design

    // Evolution variable: t = ln(mu^2 / Q0^2). t0 = 0 is the IR cutoff;
    // t_init = ln(muHard^2/Q0^2). Lambda_QCD no longer needed — Pythia's
    // AlphaStrong handles its own internal scale.
    const double Q0    = cfg_.Q0;     // set per event by caller via setQ0()
    const double Q02   = Q0 * Q0;
    const double almax = alphaS(Q02);

    double t = std::log(mu * mu / Q02);
    if (t <= 0.0) return false;

    const double kt2 = ktx * ktx + kty * kty;
    // Denominator density. Values <= 0 or < 1e-5 are treated as "no TMD
    // correction" rather than aborting the branching attempt.
    const double fB = density_(21, x, kt2, mu);
    const bool fB_ok = (fB > 1e-5);

    for (int itry = 0; itry < cfg_.maxTry; ++itry) {

        // --- z range for this trial (backward: x' = x/z <= 1  => z >= x) ---
        const double zmin = std::max(x, 1e-6);
        const double zmax = cfg_.zMax;
        if (zmin >= zmax) return false;

        // Overestimate integral  Ihat = int Phat dz / (2 C_A)
        const double I1 = std::log((1.0 - zmin) / (1.0 - zmax)); // 1/(1-z) part
        const double I2 = std::log(zmax / zmin);                 // 1/z    part
        const double Itot = I1 + I2;
        if (!(Itot > 0.0)) return false;

        // --- next (lower) scale from the Sudakov (CASCADE line 315) ---
        // CASCADE uses: T = Rsud/(weight*alp_max*Bfac) + tm_bran
        // where weight = Σ Splitt_int(i) × pdf_weight(i)
        // and pdf_weight(i) = xfb(i)/xfb(iflb) is a RATIO
        // For gluon-only: weight = Splitt_int × (fB/fB) = Splitt_int × 1
        // So TMD CANCELS in the ratio and doesn't enter here!
        const double Cmax = (almax / kTwoPi) * 2.0 * kCA * Itot * cfg_.Bfac;
        t += std::log(rand01()) / Cmax;                 // ln(R) < 0 -> t down
        if (t <= 0.0) return false;                     // fell below cutoff

        const double mu2 = Q02 * std::exp(t);
        const double muNew = std::sqrt(mu2);

        // --- splitting variable z from the overestimate shape (Eq. 35) ---
        double z;
        const double u = rand01();
        if (rand01() < I1 / Itot) {
            // sample from 1/(1-z)
            z = 1.0 - (1.0 - zmin) * std::pow((1.0 - zmax) / (1.0 - zmin), u);
        } else {
            // sample from 1/z
            z = zmin * std::pow(zmax / zmin, u);
        }
        if (z <= zmin || z >= zmax) continue;

        // --- scale-dependent soft resolution: z_max(mu) = 1 - Q0/mu -------
        // (CASCADE casbran_new.F "zm_true").  The soft-z boundary must RUN
        // with the branching scale; this is what generates the Sudakov
        // DOUBLE log (and the angular-ordering z_M independence, paper
        // Sec. 2.7).  Trials above it are vetoed, the overestimate (which
        // used the wider fixed zMax) stays a valid bound.
        const double zmRun = 1.0 - Q0 / muNew;
        if (z > zmRun) continue;

        // --- emitted parton transverse momentum, angular ordering (Eq. 41) ---
        //   mu = |q_c| / (1 - z)   =>   |q_c| = (1 - z) * mu
        const double qcMag = (1.0 - z) * muNew;
        const double phi   = kTwoPi * rand01();
        const double qcx   = qcMag * std::cos(phi);
        const double qcy   = qcMag * std::sin(phi);

        // propagating-parton transverse momentum after this branching
        const double ktxNew = ktx + qcx;
        const double ktyNew = kty + qcy;
        const double k2New  = ktxNew * ktxNew + ktyNew * ktyNew;
        const double xNew   = x / z;
        if (xNew >= 1.0) continue;

        // --- kinematic positivity veto (casbran_new.F:363-419) -----------
        // The emitted parton's beam light-cone momentum is
        //   largeLC = (1-z) * x' * sqrt(s).
        // On-shell consistency requires its other light-cone component
        //   smallLC = q_c^2 / largeLC  <=  largeLC,
        // i.e. q_c^2 <= largeLC^2. Otherwise the emission would flip into
        // the opposite hemisphere (wrong-sign pz). Reject and continue the
        // veto, exactly as CASCADE rejects q_T-vs-s violating trials.
        const double largeLC = (1.0 - z) * xNew * cfg_.rootS;
        const double qt2c    = qcx * qcx + qcy * qcy;
        if (!(largeLC > 0.0) || qt2c > largeLC * largeLC) continue;

        // --- accept/reject weight: P/Phat * alpha/alpha_max * (xf'/xf) ----
        // CASCADE line 451: wt = wtz/bfac * (xfa(IFLA)/xfb(IFLB))/pdf_weight(ifla)
        // CASCADE's casTMD returns xf (with x factor), so the ratio is:
        //   (xfa/xfb) = [x'*f(x',kt',mu)] / [x*f(x,kt,mu)] = (x'/x) * (f'/f)
        // For gluon-only: pdf_weight(0) = 1, so net weight includes (x'/x)*(f'/f)
        //
        // Our density returns f (without x), so we must add the x'/x factor:
        const double Ptrue = splittingP(21, 21, z);
        const double Phat  = 2.0 * kCA * (1.0 / (1.0 - z) + 1.0 / z);
        double w = (Ptrue / Phat) * (alphaS(mu2) / almax);

        // TMD density ratio with x-factor (CASCADE prescription).
        // If either density is missing / tiny (<=0 or < 1e-5), skip the TMD
        // ratio and keep only the CASCADE Bfac normalization.
        const double fA = density_(21, xNew, k2New, muNew);
        const bool fA_ok = (fA > 1e-5);
        if (fA_ok && fB_ok) {
            // Include (x'/x) because the callback returns f rather than x*f.
            w *= (xNew / x) * (fA / fB) / cfg_.Bfac;
        } else {
            w *= 1.0 / cfg_.Bfac;
        }
        // Note: the 1/(1-z)+1/z overestimate omits the finite
        // z(1-z) term, so w can marginally exceed 1 (<~1.06). Clamp it;
        // tighten the overestimate later if precision demands.
        if (w > 1.0) w = 1.0;

        if (rand01() < w) {
            // --- emitted-parton lab 4-momentum (approximation A3) ----------
            // It carries (1-z) of the parent's beam light-cone momentum
            // (largeLC, qt2c computed in the positivity veto above);
            // smallLC <= largeLC is guaranteed by that veto.
            const double smallLC = qt2c / largeLC;
            const double Ec  = 0.5 * (largeLC + smallLC);
            const double pzc = beamSign * 0.5 * (largeLC - smallLC);

            id  = 21;            // gluon stays a gluon (only g->gg channel)
            x   = xNew;
            ktx = ktxNew;
            kty = ktyNew;
            mu  = muNew;
            em.qx = qcx; em.qy = qcy; em.pz = pzc; em.E = Ec;
            em.z  = z;   em.mu = muNew; em.idEmit = 21;
            return true;
        }
        // rejected: continue the veto from the new (lower) scale t
    }
    return false;   // ran out of tries -> treat as no further branching
}

// ----------------------------------------------------------------------------
// Top-level backward evolution (cascps.F).
// ----------------------------------------------------------------------------
Result TMDShower::run(int idIn, double x, double ktx, double kty,
                       double muHard, int beamSign)
{
    Result r;
    r.idEnd = idIn;
    r.xEnd  = x;

    if (!cfg_.enabled) { r.ok = true; return r; }  // master off -> zero recoil
    if (idIn != 21)    { r.ok = true; return r; }  // gluon leg only (by design)
    if (!(x > 0.0) || x >= cfg_.xMax) { r.ok = true; return r; }
    if (!(muHard > 0.0)) { r.ok = true; return r; }

    int    id  = idIn;
    double xx  = x;
    double kx  = ktx;
    double ky  = kty;
    double mu  = muHard;

    double sumQx = 0.0, sumQy = 0.0;

    for (int nb = 0; nb < cfg_.maxBranch; ++nb) {
        if (xx >= cfg_.xMax) break;

        Emission em;
        if (!oneBranching(id, xx, kx, ky, mu, beamSign, em)) break;

        sumQx += em.qx;
        sumQy += em.qy;
        r.chain.push_back(em);
        ++r.nBranch;
    }

    // Net recoil delivered into the hard system: momentum conservation
    // against the emitted partons,  Q = - sum_c q_c   (cf. Eq. 36).
    r.Qx    = -sumQx;
    r.Qy    = -sumQy;
    r.xEnd  = xx;
    r.idEnd = id;
    r.ok    = true;
    return r;
}

} // namespace tmdshower
