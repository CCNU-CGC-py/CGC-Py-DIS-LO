#ifndef QS2_INTERPOLATOR_HPP
#define QS2_INTERPOLATOR_HPP

#include <string>
#include <vector>

// Linear interpolation of Q_s^2 as a function of log(x_g). Queries outside
// the tabulated x_g range are clamped to the corresponding endpoint value.
class Qs2Interpolator {
public:
    // Load a two-column table: x_g, Q_s^2 [GeV^2].
    bool loadData(const std::string& filename);

    // Return Q_s^2(x_g) [GeV^2].
    double evaluate(double xg) const;

    bool isInitialized() const { return initialized_; }
    double minXg() const { return xg_min_; }
    double maxXg() const { return xg_max_; }

private:
    bool initialized_ = false;
    std::vector<double> log_xg_grid_;
    std::vector<double> qs2_grid_;
    double xg_min_ = 0.0;
    double xg_max_ = 0.0;
};

#endif // QS2_INTERPOLATOR_HPP
