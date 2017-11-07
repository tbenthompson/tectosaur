#pragma once

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

