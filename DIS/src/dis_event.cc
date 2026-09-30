// ============================================================================
// dis_event.cc -- definitions for the DIS_gen run config + event helpers.
// ============================================================================
#include "dis_event.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

// ----- Global definitions ---------------------------------------------------
// Photon parameters.  Q2 and y can be fixed through the legacy `Q2`/`y`
// config keys, or sampled uniformly through `Q2_min`/`Q2_max` and
// `y_min`/`y_max`.  W2 is derived from Q2, y, and beam energies:
//   s - M^2 = 2 E_l (E_N + sqrt(E_N^2 - M^2))      (massless lepton, exact in M)
//   W^2     = y (s - M^2) - Q^2 + M^2
double Q2_fixed  = 4.0;
double y_fixed   = 0.7;
double Q2_min     = 4.0;
double Q2_max     = 4.0;
double y_min      = 0.7;
double y_max      = 0.7;
double W2_fixed   = 0.0;   // representative midpoint value; do not set directly
double W2_min     = 0.0;
double W2_max     = 0.0;

// Sampled phase-space ranges.
double z_reg     = 0.05;
double k_max     = 10.0;
int    nEvents_target = 30000;
int    back_to_back = 1;
int    non_back_to_back = 1;
int    shower_on = 0;
double in_table_q_max = 5.0;
double in_table_eps_f2_max = 16.0;
double in_table_lr_max = 10.0;
int    independent_fragmentation = 1;
double soft_partner_E = 2.0;

int    output_frame = 0;        // 0 = lab, 1 = gamma*-N CM
double E_lepton     = 20.0;     // GeV, electron beam
double E_nucleon    = 100.0;    // GeV, per-nucleon target beam

const double TwoPi = 2.0 * M_PI;

// PB-TMD shower handle (constructed lazily in main if TMD_SHOWER=1).
std::unique_ptr<tmdshower::TMDShower> gShower;

double dis_const_density(int, double, double, double) { return 1.0; }

double derive_W2(double Q2, double y) {
    const double M2  = M_proton * M_proton;
    const double p_N = std::sqrt(std::max(E_nucleon * E_nucleon - M2, 0.0));
    const double sM2 = 2.0 * E_lepton * (E_nucleon + p_N);   // = s - M^2
    return y * sM2 - Q2 + M2;
}

double derive_y(double Q2, double W2) {
    const double M2  = M_proton * M_proton;
    const double p_N = std::sqrt(std::max(E_nucleon * E_nucleon - M2, 0.0));
    const double sM2 = 2.0 * E_lepton * (E_nucleon + p_N);
    return (W2 + Q2 - M2) / sM2;
}

double derive_xB(double Q2, double W2) {
    const double denominator = W2 + Q2 - M_proton * M_proton;
    return denominator > 0.0 ? Q2 / denominator : -1.0;
}

// ----- Config loader --------------------------------------------------------
// Simple `key = value` text format; '#' starts a comment.
// Unknown keys / parse errors are silently ignored.
void load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[config] " << path
                  << " not found; using built-in defaults\n";
        return;
    }
    std::map<std::string, double> kv;
    std::string line;
    auto trim = [](std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
    };
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        trim(key); trim(val);
        if (key.empty() || val.empty()) continue;
        try { kv[key] = std::stod(val); } catch (...) { continue; }
    }
    auto setd = [&](const std::string& k, double& v) {
        auto it = kv.find(k); if (it != kv.end()) v = it->second;
    };
    auto seti = [&](const std::string& k, int& v) {
        auto it = kv.find(k); if (it != kv.end()) v = static_cast<int>(it->second);
    };
    setd("Q2",        Q2_fixed);
    setd("y",         y_fixed);
    Q2_min = Q2_fixed;
    Q2_max = Q2_fixed;
    y_min  = y_fixed;
    y_max  = y_fixed;
    setd("Q2_min",    Q2_min);
    setd("Q2_max",    Q2_max);
    setd("y_min",     y_min);
    setd("y_max",     y_max);
    setd("z_reg",     z_reg);
    setd("k_max",     k_max);
    seti("nEvents",      nEvents_target);
    seti("back_to_back", back_to_back);
    seti("non_back_to_back", non_back_to_back);
    seti("shower_on",    shower_on);
    setd("in_table_q_max", in_table_q_max);
    setd("in_table_eps_f2_max", in_table_eps_f2_max);
    setd("in_table_lr_max", in_table_lr_max);
    seti("independent_fragmentation", independent_fragmentation);
    setd("soft_partner_E",            soft_partner_E);
    seti("output_frame", output_frame);
    setd("E_lepton",     E_lepton);
    setd("E_nucleon",    E_nucleon);

    if (Q2_min > Q2_max) std::swap(Q2_min, Q2_max);
    if (y_min > y_max) std::swap(y_min, y_max);
    Q2_fixed = 0.5 * (Q2_min + Q2_max);
    y_fixed  = 0.5 * (y_min + y_max);

    // Derive W2 extrema over the configured rectangular Q2/y range.  W2 grows
    // with y and decreases with Q2, so these two corners are sufficient.
    W2_fixed = derive_W2(Q2_fixed, y_fixed);
    W2_min   = derive_W2(Q2_max, y_min);
    W2_max   = derive_W2(Q2_min, y_max);

    std::cout << "[config] " << path << "  "
              << "Q2_range=[" << Q2_min << ", " << Q2_max << "]"
              << " y_range=[" << y_min << ", " << y_max << "]"
              << " W2_range=[" << W2_min << ", " << W2_max << "]"
              << " z_reg=" << z_reg
              << " k_max=" << k_max
              << " nEvents=" << nEvents_target
              << " back_to_back=" << back_to_back
              << " non_back_to_back=" << non_back_to_back
              << " shower_on=" << shower_on
              << " table_q_max=" << in_table_q_max
              << " table_eps_f2_max=" << in_table_eps_f2_max
              << " table_lr_max=" << in_table_lr_max
              << " indep_frag=" << independent_fragmentation
              << " soft_E=" << soft_partner_E
              << " frame=" << (output_frame == 0 ? "nucleus-rest" : "gamma*-N CM")
              << " E_lepton=" << E_lepton << " E_N=" << E_nucleon
              << "\n";
}

// ----- Recover (k1, k2) 4-momenta -----------------------------------------
// Convention: photon along +z, target along -z. Light-cone: p^+ = E + p_z.
// The qq~ pair carries the photon's plus-momentum (eikonal), then:
//   k_i^+ = z_i * p_+^total ;  k_i^- = k_i_perp^2 / k_i^+   (massless quarks)
//
// q^+ is not Lorentz invariant -- it depends on the frame. output_frame
// selects which frame the partons are written in, and which exact form of
// q^+ is used:
//
//   output_frame = 0 (nucleus rest frame): nucleon at rest, photon along +z.
//        nu        = (W^2 + Q^2 - M^2) / (2 M)        (photon energy)
//        |q_z|     = sqrt(nu^2 + Q^2)                 (since q^2 = -Q^2)
//        q^+       = nu + |q_z|
//
//   output_frame = 1 (gamma*-N CM): both M and Q^2 enter at leading order.
//        E_gamma   = (W^2 - Q^2 - M^2) / (2 W)
//        |p_gamma| = sqrt(E_gamma^2 + Q^2)            (since q^2 = -Q^2)
//        q^+       = E_gamma + |p_gamma|
//
// Both boosts are along +z, so (k_1x, k_1y, k_2x, k_2y) pass through
// unchanged; only (p_z, E) of the partons differ between frames.
void reconstruct_partons(const DISKinematics& k,
                          Vec4Tuple& k1, Vec4Tuple& k2) {
    double p_plus_total;
    if (output_frame == 1) {
        // gamma*-N CM frame.
        const double M2      = M_proton * M_proton;
        const double W       = std::sqrt(k.W2);
        const double E_gamma = (k.W2 - k.Q2 - M2) / (2.0 * W);
        const double p_gamma = std::sqrt(E_gamma * E_gamma + k.Q2);
        p_plus_total = E_gamma + p_gamma;
    } else {
        // Nucleus rest frame, photon along +z, nucleon (M, 0).
        const double M2     = M_proton * M_proton;
        const double nu     = (k.W2 + k.Q2 - M2) / (2.0 * M_proton);
        const double q_zmag = std::sqrt(nu * nu + k.Q2);
        p_plus_total = nu + q_zmag;

        // ---- old collinear-lepton "lab" formula (commented out 2026-05-22) --
        // Treats the photon as collinear with the lepton beam (+z), uses
        //   p_plus_total = 2 y E_lepton
        // Exact in q^- -> 0; ignores the photon's ~sqrt((1-y)Q^2) transverse
        // momentum w.r.t. the lepton axis. Kept for reference; switch back by
        // restoring the line below.
        //   p_plus_total = 2.0 * y_fixed * E_lepton;
    }

    double k1_perp2 = k.k1x * k.k1x + k.k1y * k.k1y;
    double k2_perp2 = k.k2x * k.k2x + k.k2y * k.k2y;

    // On-shell condition k_i^2 = m_q^2:  k_i^- = (k_perp^2 + m_q^2) / k_i^+.
    // k.mq2 defaults to 0 (massless u, d, s); heavy-quark mode sets it per
    // event from the flavor pick.
    double k1_plus  = k.z         * p_plus_total;
    double k2_plus  = (1.0 - k.z) * p_plus_total;
    double k1_minus = (k1_plus > 0) ? (k1_perp2 + k.mq2) / k1_plus : 0.0;
    double k2_minus = (k2_plus > 0) ? (k2_perp2 + k.mq2) / k2_plus : 0.0;

    k1.pz = 0.5 * (k1_plus - k1_minus);
    k1.E  = 0.5 * (k1_plus + k1_minus);
    k2.pz = 0.5 * (k2_plus - k2_minus);
    k2.E  = 0.5 * (k2_plus + k2_minus);
    k1.px = k.k1x;  k1.py = k.k1y;
    k2.px = k.k2x;  k2.py = k.k2y;
}

// ----- Uniform draw of (Q2, y, z, k1x, k1y, k2x, k2y) for a given polarization.
// W2 is derived from the sampled (Q2, y) and the configured beam energies.
// |k1|, |k2| each restricted to [0, k_max] disk (no lower bound); sampled by
// accept-reject on the bounding square. In back-to-back mode only k1 is drawn,
// and k2 is its exact additive inverse.
std::array<double, 8>
draw_uniform(int pol, std::mt19937& gen,
              std::uniform_real_distribution<double>& U01,
              bool back) {
    auto U = [&](double lo, double hi) { return lo + (hi - lo) * U01(gen); };
    const double Q2 = U(Q2_min, Q2_max);
    const double y  = U(y_min, y_max);
    const double W2 = derive_W2(Q2, y);
    double k1x, k1y, k1m;
    do { k1x = U(-k_max, k_max); k1y = U(-k_max, k_max);
         k1m = std::sqrt(k1x*k1x + k1y*k1y); }
    while (k1m > k_max);
    double k2x = -k1x;
    double k2y = -k1y;
    if (!back) {
        double k2m;
        do { k2x = U(-k_max, k_max); k2y = U(-k_max, k_max);
             k2m = std::sqrt(k2x*k2x + k2y*k2y); }
        while (k2m > k_max);
    }
    return {
        double(pol),
        Q2,
        W2,
        U(z_reg, 1.0 - z_reg),
        k1x, k1y, k2x, k2y
    };
}
