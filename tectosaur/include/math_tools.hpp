#pragma once
#include <array>
#include <cmath>

int positive_mod(int i, int n) {
    return (i % n + n) % n;
}

template <size_t dim>
std::array<int,dim> rotation_idxs(int clicks) {
    std::array<int,dim> rotated;
    for (size_t i = 0; i < dim; i++) {
        rotated[i] = positive_mod(clicks + i, dim);
    }
    return rotated;
}

double from_interval(double a, double b, double x) {
    return ((x - a) / (b - a)) * 2.0 - 1.0;
}

std::array<double,3> linear_basis_tri(double x, double y) {
    return {1 - x - y, x, y};
}

double lawcos(double a, double b, double c) {
    return acos((a*a + b*b - c*c) / (2*a*b));
}

double rad2deg(double radians) {
    return radians * 180 / M_PI;
}

double deg2rad(double degrees) {
    return degrees / 180 * M_PI;
}
