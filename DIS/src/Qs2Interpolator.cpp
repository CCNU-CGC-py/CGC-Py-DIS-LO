#include "Qs2Interpolator.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

bool Qs2Interpolator::loadData(const std::string& filename) {
    initialized_ = false;
    log_xg_grid_.clear();
    qs2_grid_.clear();
    xg_min_ = 0.0;
    xg_max_ = 0.0;

    std::ifstream input(filename);
    if (!input) {
        std::cerr << "Qs2Interpolator: Cannot open file " << filename << '\n';
        return false;
    }

    std::vector<std::pair<double, double>> points;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream row(line);
        row >> std::ws;
        if (row.eof() || row.peek() == '#') continue;

        double xg = 0.0;
        double qs2 = 0.0;
        if (!(row >> xg >> qs2)) {
            std::cerr << "Qs2Interpolator: Invalid row " << line_number
                      << " in " << filename << '\n';
            return false;
        }
        if (!std::isfinite(xg) || !std::isfinite(qs2)
            || !(xg > 0.0) || !(qs2 > 0.0)) {
            std::cerr << "Qs2Interpolator: Non-positive or non-finite value at row "
                      << line_number << " in " << filename << '\n';
            return false;
        }
        points.emplace_back(xg, qs2);
    }

    if (points.size() < 2) {
        std::cerr << "Qs2Interpolator: Need at least two table rows in "
                  << filename << '\n';
        return false;
    }

    std::sort(points.begin(), points.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });

    log_xg_grid_.reserve(points.size());
    qs2_grid_.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i > 0 && !(points[i].first > points[i - 1].first)) {
            std::cerr << "Qs2Interpolator: Duplicate x_g value "
                      << points[i].first << " in " << filename << '\n';
            log_xg_grid_.clear();
            qs2_grid_.clear();
            return false;
        }
        log_xg_grid_.push_back(std::log(points[i].first));
        qs2_grid_.push_back(points[i].second);
    }

    xg_min_ = points.front().first;
    xg_max_ = points.back().first;
    initialized_ = true;

    std::cout << "Qs2Interpolator: Loaded " << points.size()
              << " points with x_g range [" << xg_min_ << ", " << xg_max_
              << "]\n";
    return true;
}

double Qs2Interpolator::evaluate(double xg) const {
    if (!initialized_) {
        throw std::logic_error("Qs2Interpolator is not initialized");
    }
    if (std::isnan(xg)) {
        throw std::invalid_argument("Qs2Interpolator received NaN x_g");
    }

    // Clamping before log() also gives the lower endpoint for x_g <= 0.
    if (xg <= xg_min_) return qs2_grid_.front();
    if (xg >= xg_max_) return qs2_grid_.back();

    const double log_xg = std::log(xg);
    const auto upper = std::lower_bound(
        log_xg_grid_.begin(), log_xg_grid_.end(), log_xg);
    const std::size_t hi = static_cast<std::size_t>(
        upper - log_xg_grid_.begin());
    const std::size_t lo = hi - 1;
    const double weight = (log_xg - log_xg_grid_[lo])
        / (log_xg_grid_[hi] - log_xg_grid_[lo]);
    return qs2_grid_[lo]
        + weight * (qs2_grid_[hi] - qs2_grid_[lo]);
}
