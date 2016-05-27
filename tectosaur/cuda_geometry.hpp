#pragma once

using Real = float;
struct Vec3 {
    float internal[3];

    __device__
    float& operator[](int idx) {
        return internal[idx];
    }

    __device__
    const float& operator[](int idx) const {
        return internal[idx];
    }
};

struct Tri {
    Vec3 internal[3];

    __device__
    Vec3& operator[](int idx) {
        return internal[idx];
    }

    __device__
    const Vec3& operator[](int idx) const {
        return internal[idx];
    }
};

__device__
Tri get_triangle(float* pts, int* tris, int tri_index) {
    Tri tri;
    for (int c = 0; c < 3; c++) {
        for (int d = 0; d < 3; d++) {
            tri[c][d] = pts[3 * tris[3 * tri_index + c] + d];
        }
    }
    return tri;
}

__device__
Vec3 cross(const Vec3& x, const Vec3& y) {
    return {
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0]
    };
}

__device__
Vec3 sub(const Vec3& x, const Vec3& y) {
    return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

__device__
Vec3 get_unscaled_normal(const Tri& tri) {
    return cross(sub(tri[2], tri[0]), sub(tri[2], tri[1]));
}

__device__
Real magnitude(const Vec3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
