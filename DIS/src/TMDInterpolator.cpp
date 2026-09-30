#include "TMDInterpolator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

constexpr std::array<char, 8> kExpectedMagic{
    'F', 'W', 'W', '3', 'D', '0', '1', '\0'};

template <typename T>
bool readScalar(std::ifstream& input, T& value) {
    return static_cast<bool>(
        input.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

bool readVector(
    std::ifstream& input,
    std::vector<double>& values,
    std::size_t count) {
    values.resize(count);
    if (count == 0) return true;
    return static_cast<bool>(input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(double))));
}

} // namespace

bool TMDInterpolator::loadData(const std::string& filename) {
    initialized_ = false;
    Nx_ = Nk_ = Nq2_ = 0;
    xB_grid_.clear();
    kT_grid_.clear();
    qs2_grid_.clear();
    eta_grid_.clear();
    log_xB_grid_.clear();
    log_kT_grid_.clear();
    values_.clear();

    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "TMDInterpolator: Cannot open file " << filename << '\n';
        return false;
    }

    std::array<char, 8> magic{};
    if (!input.read(magic.data(), static_cast<std::streamsize>(magic.size()))
        || magic != kExpectedMagic) {
        std::cerr << "TMDInterpolator: Invalid table magic in " << filename << '\n';
        return false;
    }

    std::uint64_t nx = 0;
    std::uint64_t nk = 0;
    std::uint64_t nq2 = 0;
    if (!readScalar(input, nx) || !readScalar(input, nk)
        || !readScalar(input, nq2)
        || !readScalar(input, q2_max_) || !readScalar(input, g2_)) {
        std::cerr << "TMDInterpolator: Truncated table header in " << filename << '\n';
        return false;
    }

    const std::uint64_t size_limit = static_cast<std::uint64_t>(
        std::numeric_limits<std::size_t>::max());
    if (nx < 2 || nk < 2 || nq2 < 2
        || nx > size_limit || nk > size_limit || nq2 > size_limit
        || nx > size_limit / nk || nx * nk > size_limit / nq2
        || !std::isfinite(q2_max_) || !std::isfinite(g2_)
        || !(q2_max_ > 0.0) || !(g2_ >= 0.0)) {
        std::cerr << "TMDInterpolator: Invalid dimensions or metadata in "
                  << filename << '\n';
        return false;
    }

    Nx_ = static_cast<std::size_t>(nx);
    Nk_ = static_cast<std::size_t>(nk);
    Nq2_ = static_cast<std::size_t>(nq2);
    const std::size_t value_count = Nx_ * Nk_ * Nq2_;

    if (!readVector(input, xB_grid_, Nx_)
        || !readVector(input, kT_grid_, Nk_)
        || !readVector(input, qs2_grid_, Nx_)
        || !readVector(input, eta_grid_, Nq2_)
        || !readVector(input, values_, value_count)) {
        std::cerr << "TMDInterpolator: Truncated table payload in " << filename << '\n';
        initialized_ = false;
        return false;
    }

    char extra = '\0';
    if (input.read(&extra, 1)) {
        std::cerr << "TMDInterpolator: Unexpected trailing bytes in "
                  << filename << '\n';
        return false;
    }

    if (!strictlyIncreasing(xB_grid_)
        || !strictlyIncreasing(kT_grid_)
        || !strictlyIncreasing(eta_grid_)
        || std::any_of(qs2_grid_.begin(), qs2_grid_.end(),
            [](double value) { return !std::isfinite(value) || !(value > 0.0); })
        || std::any_of(values_.begin(), values_.end(),
            [](double value) { return !std::isfinite(value); })) {
        std::cerr << "TMDInterpolator: Invalid axis or non-finite value in "
                  << filename << '\n';
        return false;
    }

    const double axis_tolerance = 64.0 * std::numeric_limits<double>::epsilon();
    if (std::abs(eta_grid_.front()) > axis_tolerance
        || std::abs(eta_grid_.back() - 1.0) > axis_tolerance
        || std::any_of(xB_grid_.begin(), xB_grid_.end(),
            [](double value) { return !std::isfinite(value) || !(value > 0.0); })
        || std::any_of(kT_grid_.begin(), kT_grid_.end(),
            [](double value) { return !std::isfinite(value) || !(value > 0.0); })
        || *std::max_element(qs2_grid_.begin(), qs2_grid_.end()) >= q2_max_) {
        std::cerr << "TMDInterpolator: Invalid physical table domain in "
                  << filename << '\n';
        return false;
    }

    log_xB_grid_.resize(Nx_);
    std::transform(
        xB_grid_.begin(), xB_grid_.end(), log_xB_grid_.begin(),
        [](double value) { return std::log(value); });
    log_kT_grid_.resize(Nk_);
    std::transform(
        kT_grid_.begin(), kT_grid_.end(), log_kT_grid_.begin(),
        [](double value) { return std::log(value); });

    xB_min_ = xB_grid_.front();
    xB_max_ = xB_grid_.back();
    kT_min_ = kT_grid_.front();
    kT_max_ = kT_grid_.back();
    initialized_ = true;

    std::cout << "TMDInterpolator: Loaded " << value_count
              << " F_WW values from " << filename << '\n'
              << "  Grid: " << Nx_ << " x " << Nk_ << " x " << Nq2_ << '\n'
              << "  x range: [" << xB_min_ << ", " << xB_max_ << "]\n"
              << "  k_T range: [" << kT_min_ << ", " << kT_max_ << "] GeV\n"
              << "  Q^2 upper bound: " << q2_max_ << " GeV^2\n"
              << "  g2: " << g2_ << '\n';
    return true;
}

bool TMDInterpolator::strictlyIncreasing(const std::vector<double>& values) {
    if (values.size() < 2) return false;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) return false;
        if (i > 0 && !(values[i] > values[i - 1])) return false;
    }
    return true;
}

std::size_t TMDInterpolator::findIndex(
    const std::vector<double>& grid,
    double value) {
    if (value <= grid.front()) return 0;
    if (value >= grid.back()) return grid.size() - 2;
    const auto upper = std::upper_bound(grid.begin(), grid.end(), value);
    return static_cast<std::size_t>(upper - grid.begin() - 1);
}

double TMDInterpolator::qs2AtLogX(double log_xB) const {
    const std::size_t ix = findIndex(log_xB_grid_, log_xB);
    const double weight = (log_xB - log_xB_grid_[ix])
        / (log_xB_grid_[ix + 1] - log_xB_grid_[ix]);
    return qs2_grid_[ix] + weight * (qs2_grid_[ix + 1] - qs2_grid_[ix]);
}

double TMDInterpolator::valueAt(
    std::size_t ix,
    std::size_t ik,
    std::size_t iq) const {
    return values_[((ix * Nk_) + ik) * Nq2_ + iq];
}

double TMDInterpolator::trilinearInterpolate(
    double log_xB,
    double log_kT,
    double eta) const {
    const std::size_t ix = findIndex(log_xB_grid_, log_xB);
    const std::size_t ik = findIndex(log_kT_grid_, log_kT);
    const std::size_t iq = findIndex(eta_grid_, eta);

    const double tx = (log_xB - log_xB_grid_[ix])
        / (log_xB_grid_[ix + 1] - log_xB_grid_[ix]);
    const double tk = (log_kT - log_kT_grid_[ik])
        / (log_kT_grid_[ik + 1] - log_kT_grid_[ik]);
    const double tq = (eta - eta_grid_[iq])
        / (eta_grid_[iq + 1] - eta_grid_[iq]);

    const double c000 = valueAt(ix, ik, iq);
    const double c001 = valueAt(ix, ik, iq + 1);
    const double c010 = valueAt(ix, ik + 1, iq);
    const double c011 = valueAt(ix, ik + 1, iq + 1);
    const double c100 = valueAt(ix + 1, ik, iq);
    const double c101 = valueAt(ix + 1, ik, iq + 1);
    const double c110 = valueAt(ix + 1, ik + 1, iq);
    const double c111 = valueAt(ix + 1, ik + 1, iq + 1);

    const double c00 = c000 + tq * (c001 - c000);
    const double c01 = c010 + tq * (c011 - c010);
    const double c10 = c100 + tq * (c101 - c100);
    const double c11 = c110 + tq * (c111 - c110);
    const double c0 = c00 + tk * (c01 - c00);
    const double c1 = c10 + tk * (c11 - c10);
    return c0 + tx * (c1 - c0);
}

double TMDInterpolator::TMD_gluon(double kT, double xB, double Q2) const {
    if (!initialized_ || !std::isfinite(kT) || !std::isfinite(xB)
        || !std::isfinite(Q2) || kT < kT_min_ || kT > kT_max_
        || xB < xB_min_ || xB > xB_max_ || !(Q2 > 0.0)) {
        return -1.0;
    }

    const double log_xB = std::log(xB);
    const double log_kT = std::log(kT);
    const double qs2 = qs2AtLogX(log_xB);
    const double scale = std::max({1.0, std::abs(qs2), std::abs(Q2)});
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (Q2 < qs2 - tolerance || Q2 > q2_max_ + tolerance) {
        return -1.0;
    }

    const double bounded_q2 = std::max(qs2, std::min(q2_max_, Q2));
    const double denominator = std::log(q2_max_ / qs2);
    if (!(denominator > 0.0)) return -1.0;
    const double eta = std::log(bounded_q2 / qs2) / denominator;
    return trilinearInterpolate(log_xB, log_kT, eta);
}
