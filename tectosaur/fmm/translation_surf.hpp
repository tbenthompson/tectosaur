#pragma once
#include "geometry.hpp"

template <size_t dim>
std::vector<std::array<double,dim>> surrounding_surface(size_t order);

template <>
inline std::vector<std::array<double,2>> surrounding_surface(size_t order) 
{
    std::vector<std::array<double,2>> pts(order);

    for (size_t i = 0; i < order; i++) {
        double theta = i * 2 * M_PI / static_cast<double>(order);
        pts[i] = {std::cos(theta), std::sin(theta)};
    }

    return pts;
}

template <>
inline std::vector<std::array<double,3>> surrounding_surface(size_t order)
{
    std::vector<std::array<double,3>> pts;
    double a = 4 * M_PI / order;
    double d = std::sqrt(a);
    auto M_theta = static_cast<size_t>(std::round(M_PI / d));
    double d_theta = M_PI / M_theta;
    double d_phi = a / d_theta;
    for (size_t m = 0; m < M_theta; m++) {
        double theta = M_PI * (m + 0.5) / M_theta;
        auto M_phi = static_cast<size_t>(
            std::round(2 * M_PI * std::sin(theta) / d_phi)
        );
        for (size_t n = 0; n < M_phi; n++) {
            double phi = 2 * M_PI * n / M_phi;
            double x = std::sin(theta) * std::cos(phi);
            double y = std::sin(theta) * std::sin(phi);
            double z = std::cos(theta);
            pts.push_back({x, y, z});
        }
    }

    return pts;
}

template <size_t dim>
std::vector<std::array<double,dim>> inscribe_surf(const Ball<dim>& b, double scaling,
                                const std::vector<std::array<double,dim>>& fmm_surf) {
    std::vector<std::array<double,dim>> out(fmm_surf.size());
    for (size_t i = 0; i < fmm_surf.size(); i++) {
        for (size_t d = 0; d < dim; d++) {
            out[i][d] = fmm_surf[i][d] * b.R * scaling + b.center[d];
        }
    }
    return out;
}

