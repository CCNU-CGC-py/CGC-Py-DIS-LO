// ============================================================================
// main.cc -- DIS dijet generator with TMD gluon distribution
//
// This version integrates the tabulated Sudakov-evolved WW distribution
// F_WW(x_B, k_T, Q^2) into the PB-TMD parton shower as the input density.
//
// PHYSICS:
//   - TMD gluon distribution: F_WW(x, k_T, Q^2)
//   - Trilinear interpolation is performed in log(x), log(k_T), and the
//     normalized log(Q^2/Q_s^2(x)) coordinate
//   - This TMD is used as the density callback in the backward shower
//   - The shower evolves from hard scale mu_hard down to IR cutoff Q0
//
// MODIFICATIONS from main.cc:
//   1. Added TMDInterpolator to load and interpolate FWW_TMD_TABLE.bin
//   2. Replaced dis_const_density with tmd_gluon_density callback
//   3. The density function now uses the shower scale Q^2 = mu^2
//
// ============================================================================

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "Pythia8/Pythia.h"

#include "dis_event.hpp"
#include "tmd_shower.hpp"
#include "TMDInterpolator.hpp"
#include "Qs2Interpolator.hpp"

using namespace std;
using namespace Pythia8;

// Global TMD interpolator (initialized in main)
static std::unique_ptr<TMDInterpolator> g_tmd_interp;
static std::unique_ptr<Qs2Interpolator> g_qs2_interp;

// TMD gluon density callback for the parton shower
// Returns F_WW(x, k_T, Q^2=mu^2)
//
// Arguments:
//   id  : PDG id (21 = gluon)
//   x   : longitudinal momentum fraction
//   kt2 : transverse momentum squared [GeV^2]
//   mu  : evolution scale [GeV]
double tmd_gluon_density(int id, double x, double kt2, double mu) {
    if (id != 21) return 0.0;  // only gluons
    if (!g_tmd_interp || !g_tmd_interp->isInitialized()) return -1.0;
    if (!(kt2 >= 0.0) || !(mu > 0.0)) return -1.0;

    double kT = std::sqrt(kt2);
    double Q2 = mu * mu;

    return g_tmd_interp->TMD_gluon(kT, x, Q2);
}

int main(int /*argc*/, char* /*argv*/[]) {

    // --- Load TMD interpolator ---
    g_tmd_interp = std::make_unique<TMDInterpolator>();
    if (!g_tmd_interp->loadData("src/FWW_TMD_TABLE.bin")) {
        cerr << "Failed to load TMD data file: src/FWW_TMD_TABLE.bin" << endl;
        return 1;
    }
    cout << "TMD gluon distribution loaded successfully.\n" << endl;

    // --- Load run config ---
    const char* config_env = std::getenv("DIS_CONFIG");
    const std::string config_path = config_env ? config_env : "src/dis_config.txt";
    load_config(config_path);

    // Q0 is set event by event from the tabulated saturation scale.
    if (shower_on) {
        g_qs2_interp = std::make_unique<Qs2Interpolator>();
        if (!g_qs2_interp->loadData("src/Qs2_table.dat")) {
            cerr << "Failed to load Q_s^2 table: src/Qs2_table.dat" << endl;
            return 1;
        }
    }

    // --- Master seed: from MC_SEED env var, else time-based. Logged below.
    // All RNGs (Pythia, std::mt19937 engine, TMD shower) derive from this one
    // master via std::seed_seq so a single value reproduces the entire run.
    unsigned master_seed;
    if (const char* s = std::getenv("MC_SEED"); s && std::atoi(s) > 0) {
        master_seed = static_cast<unsigned>(std::atoi(s));
    } else {
        master_seed = static_cast<unsigned>(
            std::chrono::system_clock::now().time_since_epoch().count());
    }
    std::cout << "Master seed: " << master_seed
              << " (set MC_SEED=<int> to reproduce)" << std::endl;
    std::seed_seq sseq{master_seed, 0xC0FFEEu};
    std::vector<unsigned> sub(3);
    sseq.generate(sub.begin(), sub.end());
    // sub[0] -> Pythia,  sub[1] -> std::mt19937 engine,  sub[2] -> TMD shower

    // --- Pythia: FSR-only afterburner ---
    const char* pythia_data_env = std::getenv("DIS_PYTHIA8DATA");
    const char* pythia_data = pythia_data_env
        ? pythia_data_env : "/home/congyi/pythia8310/share/Pythia8/xmldoc";
    // Match the XML data to the Pythia installation selected by Makefile_tmd.
    setenv("PYTHIA8DATA", pythia_data, /*overwrite=*/1);
    Pythia pythia;
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " +
                      std::to_string(sub[0] % 900000000u + 1u));

    pythia.readString("ProcessLevel:all = off");
    pythia.readString("ProcessLevel:resonanceDecays = on");
    pythia.readString("PartonLevel:FSRinResonances = on");

    if (!pythia.init()) {
        cerr << "Pythia initialization failed!" << endl;
        return 1;
    }
    pythia.readString("111:mayDecay = off");

    // --- PB-TMD shower init with TMD gluon density ---
    if (shower_on) {
        tmdshower::Config scfg;
        scfg.rootS = std::sqrt(std::max(W2_max, W2_fixed));
        // Use TMD gluon density instead of constant density
        gShower = std::make_unique<tmdshower::TMDShower>(
            scfg, &tmd_gluon_density, /*seed=*/sub[2]);
    }
    std::cout << "PB-TMD target-gluon shower: "
              << (shower_on
                      ? "ON (with TMD gluon; Q0=sqrt(Qs2(xg)) from table)"
                      : "OFF")
              << "\n"
              << "Hard-process sampling: coherent="
              << (back_to_back ? "ON" : "OFF")
              << " inelastic=" << (non_back_to_back ? "ON" : "OFF") << "\n";

    // --- RNG ---
    mt19937 engine(sub[1]);
    uniform_real_distribution<double> U01(0.0, 1.0);

    // --- Uniform quark flavor proposal ---
    // The external analysis applies e_q^2 and divides by this 1/3 proposal.
    auto pick_flavor = [&](mt19937& g) -> int {
        static constexpr int flavors[] = {1, 2, 3};
        return flavors[std::uniform_int_distribution<int>(0, 2)(g)];
    };

    // --- Output ---
    const char* output_env = std::getenv("DIS_OUTPUT");
    const std::string output_path = output_env
        ? output_env : "results/dis_dijet_tmd_output.txt";
    std::ofstream outFile(output_path);
    if (!outFile.is_open()) {
        cerr << "Cannot open " << output_path << endl;
        return 1;
    }
    outFile << std::setprecision(17);
    outFile << "# DIS dijet events, schema_version=4\n";
    outFile << "# TMD: F_WW(x, k_T, Q^2) from FWW_TMD_TABLE.bin"
               " with out-of-range value -1\n";
    outFile << "# event_semantics: quark_header=pre_shower"
               " hadrons=post_pythia_final_state\n";
    outFile << "# columns: EV pol back_to_back flavor Q2 y W2 xB xg z"
               " k1x k1y k2x k2y eta1 eta2"
               " nHad\n";
    outFile << "# sampling: Q2_min=" << Q2_min
            << " Q2_max=" << Q2_max
            << " y_min=" << y_min
            << " y_max=" << y_max
            << " z_min=" << z_reg
            << " z_max=" << (1.0 - z_reg)
            << " k_max=" << k_max
            << " flavor_pdf=" << (1.0 / 3.0)
            << " E_lepton=" << E_lepton
            << " E_nucleon=" << E_nucleon
            << " shower_on=" << shower_on
            << " in_table_q_max=" << in_table_q_max
            << " in_table_eps_f2_max=" << in_table_eps_f2_max
            << " in_table_lr_max=" << in_table_lr_max
            << " output_frame=" << output_frame
            << " independent_fragmentation=" << independent_fragmentation
            << " soft_partner_E=" << soft_partner_E << "\n";
    outFile << "# inelastic_emulator_support: xg_min=0.0000001 xg_max=0.01"
               " eps_min=0.1 eps_max=4 k1_min=0.1 k1_max=25"
               " q_min=1 q_max=10\n";
    outFile << "# then nHad lines:  id px py pz E\n";

    struct GenerationStats {
        int written = 0;
        long tries = 0;
    };
    int nFail = 0;

    auto generate_pol = [&](int pol_label, bool back) -> GenerationStats {
        const long max_tries = 100L * nEvents_target;
        GenerationStats stats;
        while (stats.written < nEvents_target && stats.tries < max_tries) {
            ++stats.tries;
            auto sample = draw_uniform(pol_label, engine, U01, back);

            DISKinematics k = compute_dis_kinematics(sample);
            if (!k.valid) continue;
            // if (!back) {
            //     const double k1_abs = std::hypot(k.k1x, k.k1y);
            //     const double eps_f = std::sqrt(k.eps_f2);
            //     const bool in_emulator_support =
            //         k.xg >= 1.0e-3 && k.xg <= 1.0e-2
            //         && eps_f >= 0.5 && eps_f <= 2.0
            //         && k1_abs >= 1.0 && k1_abs <= 10.0
            //         && k.q >= 1.0 && k.q <= 5.0;
            //     if (!in_emulator_support) continue;
            // }

            // --- Parton 4-momenta reconstruction (DIS variables -> 4-vectors)
            Vec4Tuple p1, p2;
            reconstruct_partons(k, p1, p2);
            Vec4 v1(p1.px, p1.py, p1.pz, p1.E);
            Vec4 v2(p2.px, p2.py, p2.pz, p2.E);

            // Preserve the hard-process tuple before PB-TMD recoil. The event
            // row is written only after successful hadronization, but these
            // immutable values are the quark fields serialized in that row.
            const DISKinematics hard_k = k;
            const Vec4 hard_v1 = v1;
            const Vec4 hard_v2 = v2;
            const double hard_pt1 = std::hypot(hard_v1.px(), hard_v1.py());
            const double hard_pt2 = std::hypot(hard_v2.px(), hard_v2.py());
            if (!(hard_pt1 > 0.0) || !(hard_pt2 > 0.0)) continue;
            const double hard_eta1 = std::asinh(hard_v1.pz() / hard_pt1);
            const double hard_eta2 = std::asinh(hard_v2.pz() / hard_pt2);

            // Relative dijet pT used as the hard scale for PB-TMD and Pythia FSR.
            const double Px_rel = (1.0 - k.z) * v1.px() - k.z * v2.px();
            const double Py_rel = (1.0 - k.z) * v1.py() - k.z * v2.py();
            const double P_mag  = std::sqrt(Px_rel*Px_rel + Py_rel*Py_rel);
            if (!(P_mag > 0.0)) continue;
            //const double shower_scale = P_mag/std::sqrt(k.z*(1.0-k.z));
            const double shower_scale = P_mag;

            // PB-TMD target-gluon shower with TMD density
            if (gShower) {
                const double qs2_evt = g_qs2_interp->evaluate(k.xg);
                gShower->setQ0(std::sqrt(qs2_evt));
                gShower->setRootS(std::sqrt(k.W2));

                const double qx_tot = v1.px() + v2.px();
                const double qy_tot = v1.py() + v2.py();
                auto r = gShower->run(21, k.xg, qx_tot, qy_tot, shower_scale,
                                       /*beamSign=*/-1);
                if (r.ok) {
                    tmdshower::applyRecoil(v1, v2, r.Qx, r.Qy);

                    // Ensure on-shell: recalculate E to satisfy E^2 = p^2 + m^2
                    double m1 = v1.mCalc();
                    double m2 = v2.mCalc();
                    double p1sq = v1.px()*v1.px() + v1.py()*v1.py() + v1.pz()*v1.pz();
                    double p2sq = v2.px()*v2.px() + v2.py()*v2.py() + v2.pz()*v2.pz();
                    double E1_new = std::sqrt(p1sq + m1*m1);
                    double E2_new = std::sqrt(p2sq + m2*m2);
                    v1.e(E1_new);
                    v2.e(E2_new);

                    k.k1x = v1.px();
                    k.k1y = v1.py();
                    k.k2x = v2.px();
                    k.k2y = v2.py();
                    update_derived(k);
                }
            }

            int q_id    = pick_flavor(engine);
            int qbar_id = -q_id;
            const double y_evt  = derive_y(k.Q2, k.W2);
            const double xB_evt = derive_xB(k.Q2, k.W2);
            const double pt1 = std::hypot(v1.px(), v1.py());
            const double pt2 = std::hypot(v2.px(), v2.py());
            if (!(pt1 > 0.0) || !(pt2 > 0.0)) continue;

            // Verify 4-momentum is on-shell before feeding to Pythia
            double m1_calc = v1.mCalc();
            double m2_calc = v2.mCalc();
            double tol = 0.1; // 100 MeV tolerance
            if (std::abs(m1_calc) > tol || std::abs(m2_calc) > tol) {
                // Mass should be ~0 for light quarks
                // If not, skip this event
                continue;
            }

            Event& event = pythia.event;
            event.reset();

            if (independent_fragmentation) {
                const int    soft_id   = 1 + std::uniform_int_distribution<int>(0, 2)(engine);
                const double m_soft    = 0.33;
                const double E_soft    = soft_partner_E;
                const double p_soft    = std::sqrt(
                    std::max(E_soft*E_soft - m_soft*m_soft, 0.0));
                auto softAlong = [](const Vec4& v, double p, double E) {
                    const double p_ref = std::sqrt(
                        v.px()*v.px() + v.py()*v.py() + v.pz()*v.pz());
                    if (p_ref <= 1.0e-12) return Vec4(0.0, 0.0, -p, E);
                    return Vec4(
                        p * v.px() / p_ref,
                        p * v.py() / p_ref,
                        p * v.pz() / p_ref,
                        E);
                };
                Vec4 vsoft1 = softAlong(v1, p_soft, E_soft);
                Vec4 vsoft2 = softAlong(v2, p_soft, E_soft);

                event.append(q_id,     23, 101,   0, v1,     v1.mCalc());
                event.append(qbar_id,  23,   0, 102, v2,     v2.mCalc());
                event.append(-soft_id, 23,   0, 101, vsoft1, m_soft);
                event.append( soft_id, 23, 102,   0, vsoft2, m_soft);
            } else {
                event.append(q_id,    23, 101,   0, v1, v1.mCalc());
                event.append(qbar_id, 23,   0, 101, v2, v2.mCalc());
            }

            
            event[1].scale(shower_scale);
            event[2].scale(shower_scale);
            pythia.forceTimeShower(1, 2, shower_scale);

            if (!pythia.next()) { nFail++; continue; }

            int nHad = 0;
            for (int j = 0; j < event.size(); ++j)
                if (event[j].isFinal() && event[j].isHadron()) nHad++;

            outFile << "EV " << pol_label
                    << " " << static_cast<int>(back)
                    << " " << q_id
                    << " " << k.Q2 << " " << y_evt << " " << k.W2
                    << " " << xB_evt << " " << hard_k.xg << " " << hard_k.z
                    << " " << hard_k.k1x << " " << hard_k.k1y
                    << " " << hard_k.k2x << " " << hard_k.k2y
                    << " " << hard_eta1 << " " << hard_eta2
                    << " " << nHad << "\n";
            for (int j = 0; j < event.size(); ++j) {
                if (!event[j].isFinal() || !event[j].isHadron()) continue;
                outFile << event[j].id() << " "
                        << event[j].px() << " " << event[j].py() << " "
                        << event[j].pz() << " " << event[j].e()  << "\n";
            }
            ++stats.written;
        }
        if (stats.written < nEvents_target)
            cerr << "[pol=" << pol_label << "] hit max_tries with only "
                 << stats.written << " events" << endl;
        return stats;
    };

    GenerationStats inelT, inelL, cohT, cohL;
    if (non_back_to_back) {
        inelT = generate_pol(0, false);
        inelL = generate_pol(1, false);
    }
    if (back_to_back) {
        cohT = generate_pol(0, true);
        cohL = generate_pol(1, true);
    }

    auto write_generation_stats = [&](const char* mode, int pol,
                                      const GenerationStats& stats) {
        outFile << "# generation: mode=" << mode
                << " pol=" << pol
                << " written=" << stats.written
                << " tries=" << stats.tries << "\n";
    };
    if (non_back_to_back) {
        write_generation_stats("inelastic", 0, inelT);
        write_generation_stats("inelastic", 1, inelL);
    }
    if (back_to_back) {
        write_generation_stats("coherent", 0, cohT);
        write_generation_stats("coherent", 1, cohL);
    }

    outFile.close();
    pythia.stat();

    cout << "\nDIS dijet generation summary (TMD version):" << endl;
    cout << "  Inelastic transverse: " << inelT.written
         << " / " << inelT.tries << " proposals" << endl;
    cout << "  Inelastic longitudinal: " << inelL.written
         << " / " << inelL.tries << " proposals" << endl;
    cout << "  Coherent transverse: " << cohT.written
         << " / " << cohT.tries << " proposals" << endl;
    cout << "  Coherent longitudinal: " << cohL.written
         << " / " << cohL.tries << " proposals" << endl;
    cout << "  Failures:                    " << nFail << endl;
    cout << "  Output: " << output_path << endl;

    return 0;
}
