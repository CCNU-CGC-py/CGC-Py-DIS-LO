// ============================================================================
// dis_tables_stub.cc
//
// Implementations of the table interfaces declared extern in dis_kinematics.hpp:
//
//   dipole_FT(ell, Y)                      -- GBW closed form, analytic
//   quadrupole_inel_T(lr, q, eps_f2, Y)    -- trilinear interp of I_T.dat
//   quadrupole_inel_L(lr, q, eps_f2, Y)    -- trilinear interp of I_L.dat
//
// Tables are loaded from data/I_T.dat / data/I_L.dat on first call. Generated
// by scripts/dis_julia/run_dis_tables.jl at Q_s^2 = 1 GeV^2 (fixed). Y is
// ignored: Q_s is not evolved in this build.
//
// File format expected (3-D table, eps_f varies fastest, then q, then l_r):
//   # header line
//   l_r  q  eps_f  I_inel
//   ...
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "dis_kinematics.hpp"

namespace {

// Q_s^2 the tables were generated at. Hardcoded to match run_dis_tables.jl.
const double Qs2_table = 1.0;

struct GBWTable {
    std::vector<double> lr_grid;
    std::vector<double> q_grid;
    std::vector<double> ef_grid;
    std::vector<double> data;    // size = N_lr * N_q * N_ef, lr-major
    int N_lr = 0, N_q = 0, N_ef = 0;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line;
        std::getline(f, line);    // skip header
        std::vector<double> ls, qs, es, vs;
        double l, q, e, v;
        while (f >> l >> q >> e >> v) {
            ls.push_back(l); qs.push_back(q); es.push_back(e); vs.push_back(v);
        }
        if (vs.empty()) return false;

        auto uniq = [](const std::vector<double>& src) {
            std::vector<double> u(src);
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end(), [](double a, double b) {
                return std::abs(a - b) < 1e-9 * (1.0 + std::abs(a));
            }), u.end());
            return u;
        };
        lr_grid = uniq(ls);
        q_grid  = uniq(qs);
        ef_grid = uniq(es);
        N_lr = (int)lr_grid.size();
        N_q  = (int)q_grid.size();
        N_ef = (int)ef_grid.size();

        if ((int)vs.size() != N_lr * N_q * N_ef) {
            std::cerr << "Table " << path << ": row count " << vs.size()
                      << " != " << N_lr << "*" << N_q << "*" << N_ef << "\n";
            return false;
        }
        data = std::move(vs);
        return true;
    }

    double eval(double lr, double q, double ef) const {
        if (N_lr == 0) return 0.0;
        auto find_idx = [](const std::vector<double>& g, double x,
                            int& i, double& t) {
            int n = (int)g.size();
            if (x <= g.front())   { i = 0;     t = 0.0; return; }
            if (x >= g.back())    { i = n - 2; t = 1.0; return; }
            int lo = 0, hi = n - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (g[mid] <= x) lo = mid; else hi = mid;
            }
            i = lo;
            t = (x - g[i]) / (g[i + 1] - g[i]);
        };
        int i, j, k;
        double tx, ty, tz;
        find_idx(lr_grid, lr, i, tx);
        find_idx(q_grid,  q,  j, ty);
        find_idx(ef_grid, ef, k, tz);

        auto at = [&](int a, int b, int c) {
            return data[(a * N_q + b) * N_ef + c];
        };
        double c00 = at(i,   j,   k  ) * (1 - tx) + at(i+1, j,   k  ) * tx;
        double c01 = at(i,   j,   k+1) * (1 - tx) + at(i+1, j,   k+1) * tx;
        double c10 = at(i,   j+1, k  ) * (1 - tx) + at(i+1, j+1, k  ) * tx;
        double c11 = at(i,   j+1, k+1) * (1 - tx) + at(i+1, j+1, k+1) * tx;
        double c0  = c00 * (1 - ty) + c10 * ty;
        double c1  = c01 * (1 - ty) + c11 * ty;
        return c0 * (1 - tz) + c1 * tz;
    }
};

GBWTable g_T;
GBWTable g_L;
bool     g_loaded = false;

void ensure_loaded() {
    if (g_loaded) return;
    if (!g_T.load("data/I_T.dat")) {
        std::cerr << "FATAL: cannot read data/I_T.dat (run from DIS_gen/).\n";
        std::exit(1);
    }
    if (!g_L.load("data/I_L.dat")) {
        std::cerr << "FATAL: cannot read data/I_L.dat (run from DIS_gen/).\n";
        std::exit(1);
    }
    std::cout << "Loaded I_T.dat (" << g_T.N_lr << " x " << g_T.N_q
              << " x " << g_T.N_ef << ")  and  I_L.dat ("
              << g_L.N_lr << " x " << g_L.N_q << " x " << g_L.N_ef << ")\n";
    g_loaded = true;
}

} // namespace

// GBW closed-form dipole FT — analytic, no table needed.
double dipole_FT(double ell, double /*Y*/) {
    return (4.0 * M_PI / Qs2_table) * std::exp(-(ell * ell) / Qs2_table);
}

double quadrupole_inel_T(double lr, double q, double eps_f2, double /*Y*/) {
    ensure_loaded();
    double ef = std::sqrt(std::max(eps_f2, 0.0));
    return g_T.eval(lr, q, ef);
}

double quadrupole_inel_L(double lr, double q, double eps_f2, double /*Y*/) {
    ensure_loaded();
    double ef = std::sqrt(std::max(eps_f2, 0.0));
    return g_L.eval(lr, q, ef);
}
