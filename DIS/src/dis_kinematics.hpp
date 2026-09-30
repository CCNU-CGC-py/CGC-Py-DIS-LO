// ============================================================================
// dis_kinematics.hpp
//
// DIS dijet kinematics for gamma*A -> q qbar in the large-Nc CGC framework
// (Dominguez-Marquet-Xiao-Yuan, PRD 83, 105005).
//
// Per-event sampling array layout (8 doubles):
//   {pol, Q2, W2, z, k1x, k1y, k2x, k2y}
//
//   pol = 0 (transverse) or 1 (longitudinal)
//   (k1, k2) are the q, qbar transverse momenta directly (no P/q detour).
//
// Derived quantities used by the cross-section table:
//   lr = |k1 - k2| / 2          (relative momentum magnitude entering I^{T,L})
//   q  = |k1 + k2|              (qqbar transverse imbalance magnitude)
//
// Table interfaces (dipole_FT, quadrupole_inel_*) are declared extern here
// and must be supplied externally.
// ============================================================================
#ifndef DIS_KINEMATICS_HPP
#define DIS_KINEMATICS_HPP

#include <cmath>
#include <array>

// Named indices for the 8-element DIS sampling array
enum {
    DIS_POL = 0,
    DIS_Q2  = 1,
    DIS_W2  = 2,
    DIS_Z   = 3,
    DIS_K1X = 4,
    DIS_K1Y = 5,
    DIS_K2X = 6,
    DIS_K2Y = 7
};

// Physical constants
const double Nc_DIS    = 3.0;
const double alpha_em  = 1.0 / 137.0;
const double eq2_uds   = 4./9. + 1./9. + 1./9.;   // sum_q e_q^2 for u,d,s
const double M_proton  = 0.938;
const double x0_dipole = 0.01;                  // dipole evolution start

// Configurable support limits applied by compute_dis_kinematics().
extern double in_table_q_max;
extern double in_table_eps_f2_max;
extern double in_table_lr_max;

// ============================================================================
// Kinematics: derived quantities from the sampled 8-tuple
// ============================================================================
struct DISKinematics {
    double Q2, W2, z;
    double k1x, k1y;             // quark transverse momentum
    double k2x, k2y;             // antiquark transverse momentum
    double lr;                   // |k1 - k2| / 2
    double q;                    // |k1 + k2|
    double eps_f2;               // z(1-z) Q^2  (+ m_q^2 once heavy quark is on)
    double mq2 = 0.0;            // quark mass squared; 0 for u, d, s
    double xg;                   // dipole-evolution gluon-x (DMXY)
    double Y;                    // ln(x0 / xg)
    int    pol;                  // 0 = transverse, 1 = longitudinal
    bool   valid;                // false if kinematically forbidden
};

// Recompute lr, q, xg, Y from (k1, k2, z, Q2, W2). Used in compute_dis_kinematics
// and after any in-place modification of (k1, k2) (e.g. shower recoil).
inline void update_derived(DISKinematics& k) {
    double dx = k.k1x - k.k2x;
    double dy = k.k1y - k.k2y;
    k.lr = 0.5 * std::sqrt(dx*dx + dy*dy);

    double sx = k.k1x + k.k2x;
    double sy = k.k1y + k.k2y;
    k.q  = std::sqrt(sx*sx + sy*sy);

    double k1p2 = k.k1x*k.k1x + k.k1y*k.k1y;
    double k2p2 = k.k2x*k.k2x + k.k2y*k.k2y;

    // xg = (Q^2 + M_qqbar^2) / (W^2 + Q^2 - M^2)
    // M_qqbar^2 = |k1|^2/z + |k2|^2/(1-z) for massless q,qbar with
    // lightcone fractions z and 1-z (algebraically equivalent to the
    // older Q^2 + q^2 + P^2/(z(1-z)) form).
    double denom = k.W2 + k.Q2 - M_proton * M_proton;
    if (denom > 0.0 && k.z > 0.0 && k.z < 1.0) {
        k.xg = (k.Q2 + k1p2 / k.z + k2p2 / (1.0 - k.z)) / denom;
        k.Y  = (k.xg > 0.0) ? std::log(x0_dipole / k.xg) : -1.0;
    } else {
        k.xg = -1.0;
        k.Y  = -1.0;
    }
}

inline DISKinematics compute_dis_kinematics(const std::array<double, 8>& p) {
    DISKinematics k;
    k.pol = static_cast<int>(p[DIS_POL]);
    k.Q2  = p[DIS_Q2];
    k.W2  = p[DIS_W2];
    k.z   = p[DIS_Z];
    k.k1x = p[DIS_K1X];
    k.k1y = p[DIS_K1Y];
    k.k2x = p[DIS_K2X];
    k.k2y = p[DIS_K2Y];

    k.eps_f2 = k.z * (1.0 - k.z) * k.Q2;
    update_derived(k);

    // Upper support limits are configured in dis_config.txt. There is no
    // lower-bound rejection.
    bool in_table_support = (k.q <= in_table_q_max)
                         && (k.eps_f2 <= in_table_eps_f2_max)
                         && (k.lr <= in_table_lr_max);

    k.valid = (k.eps_f2 > 0.0)
           && (k.xg > 0.0) && (k.xg < 1.0)
           && (k.z > 0.0) && (k.z < 1.0)
           && in_table_support;
    return k;
}

// ============================================================================
// External table interfaces — supplied by the user.
//
//   dipole_FT(ell, Y) :       S~(ell, Y), used for the elastic dressed wave
//                              function (delta(q_perp) normalization piece).
//
//   quadrupole_inel_T(lr, q, eps_f2, Y) :  inelastic 6D integral, transverse.
//   quadrupole_inel_L(lr, q, eps_f2, Y) :  inelastic 6D integral, longitudinal.
//
// Units: GeV throughout. Y is dimensionless.
// ============================================================================
extern double dipole_FT(double ell, double Y);

extern double quadrupole_inel_T(double lr, double q, double eps_f2, double Y);
extern double quadrupole_inel_L(double lr, double q, double eps_f2, double Y);

#endif // DIS_KINEMATICS_HPP
