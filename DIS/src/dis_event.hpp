// ============================================================================
// dis_event.hpp -- run config + event-construction helpers for DIS_gen.
// (Everything that used to live in DIS_dijet_MC.cc apart from main().)
// ============================================================================
#ifndef DIS_EVENT_HPP
#define DIS_EVENT_HPP

#include <array>
#include <memory>
#include <random>
#include <string>

#include "dis_kinematics.hpp"
#include "tmd_shower.hpp"

// ----- Run configuration globals (defaults overwritten by dis_config.txt) ---
extern double Q2_fixed;         // backward-compatible fixed value
extern double y_fixed;          // backward-compatible fixed value
extern double Q2_min, Q2_max;   // sampled photon virtuality range [GeV^2]
extern double y_min, y_max;     // sampled DIS inelasticity range
extern double W2_fixed;         // representative value at midpoint ranges
extern double W2_min, W2_max;   // extrema from configured Q2/y ranges
extern double z_reg;
extern double k_max;
extern int    nEvents_target;
extern int    back_to_back;     // 1 = generate the coherent q_perp = 0 strata
extern int    non_back_to_back; // 1 = generate the inelastic q_perp != 0 strata
extern int    shower_on;        // 1 = enable PB-TMD shower, 0 = disable
extern double in_table_q_max;
extern double in_table_eps_f2_max;
extern double in_table_lr_max;

// Independent fragmentation:
//   0 = q and qbar share one color line (single Lund string between them).
//   1 = each hard parton gets its own soft target-collinear color partner,
//       so the two hadronize on independent strings -- closer to the CGC
//       picture where each parton's color is connected to the target remnant.
extern int    independent_fragmentation;
extern double soft_partner_E;   // GeV, target-collinear soft-partner energy

// Output frame for the reconstructed partons:
//   0 = target rest frame [default],   1 = gamma*-N CM.
extern int    output_frame;
extern double E_lepton;         // lab-frame lepton beam energy [GeV]
extern double E_nucleon;        // lab-frame per-nucleon target beam energy

extern const double TwoPi;

// ----- PB-TMD target-gluon shower handle (null = OFF) -----------------------
extern std::unique_ptr<tmdshower::TMDShower> gShower;

// Density callback used when the shower is enabled (constant: Stage A).
double dis_const_density(int, double, double, double);

// ----- Parton 4-momentum tuple -----
struct Vec4Tuple { double px, py, pz, E; };

// ----- Helpers --------------------------------------------------------------
void load_config(const std::string& path);

double derive_W2(double Q2, double y);

double derive_y(double Q2, double W2);

double derive_xB(double Q2, double W2);

void reconstruct_partons(const DISKinematics& k,
                          Vec4Tuple& k1, Vec4Tuple& k2);

std::array<double, 8>
draw_uniform(int pol, std::mt19937& gen,
              std::uniform_real_distribution<double>& U01,
              bool back);

#endif // DIS_EVENT_HPP
