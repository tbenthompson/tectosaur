#pragma once
#include <array>

using Vec2 = std::array<double,2>;
using Vec3 = std::array<double,3>;
using Tensor3 = std::array<Vec3,3>;

inline Tensor3 transpose(const Tensor3& A) {
    Tensor3 out;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1][d2] = A[d2][d1];
        }
    }
    return out;
}

inline Tensor3 mult(const Tensor3& A, double factor) {
    Tensor3 out;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1][d2] = A[d1][d2] * factor;
        }
    }
    return out;
}

inline Tensor3 mult(const Tensor3& A, const Tensor3& B) {
    Tensor3 out{};
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1][d2] = 0.0;
            for (int d3 = 0; d3 < 3; d3++) {
                out[d1][d2] += A[d1][d3] * B[d3][d2]; 
            }
        }
    }
    return out;
}

inline Tensor3 outer(const Vec3& A, const Vec3& B) {
    Tensor3 out;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1][d2] = A[d1] * B[d2];
        }
    }
    return out;
}

inline Tensor3 rotation_matrix(const Vec3& axis, double theta) {
    Tensor3 cross_mat = {{
        {0, -axis[2], axis[1]},
        {axis[2], 0, -axis[0]},
        {-axis[1], axis[0], 0}
    }};
    Tensor3 outer_mat = outer(axis, axis);

    Tensor3 out;    
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1][d2] =
                cos(theta) * ((d1 == d2) ? 1 : 0) +
                sin(theta) * cross_mat[d1][d2] +
                (1 - cos(theta)) * outer_mat[d1][d2];
        }
    }
    return out;
}

inline Vec3 mult(const Vec3& A, double f) {
    return {A[0] * f, A[1] * f, A[2] * f};
}

inline Vec3 div(const Vec3& A, double f) {
    double multfactor = 1.0 / f;
    return mult(A, multfactor);
}

inline Vec3 sub(const Vec3& A, const Vec3& B) {
    return {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
}

inline Vec3 add(const Vec3& A, const Vec3& B) {
    return {A[0] + B[0], A[1] + B[1], A[2] + B[2]};
}

inline double dot(const Vec3& A, const Vec3& B) {
    return A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
}

inline double length(const Vec3& A) {
    return std::sqrt(dot(A, A));
}

inline Vec3 projection(const Vec3& V, const Vec3& b) {
    return mult(b, dot(V, b) / dot(b, b));
}

inline Vec3 cross(const Vec3& x, const Vec3& y) {
    return {
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0]
    };
}

inline Vec3 tri_normal(const Tensor3& t) {
    return cross(sub(t[2], t[0]), sub(t[2], t[1]));
}

inline Vec3 triangle_internal_angles(const Tensor3& tri) {
    auto v01 = sub(tri[1], tri[0]);
    auto v02 = sub(tri[2], tri[0]);
    auto v12 = sub(tri[2], tri[1]);

    auto L01 = length(v01);
    auto L02 = length(v02);
    auto L12 = length(v12);

    auto A1 = acos(dot(v01, v02) / (L01 * L02));
    auto A2 = acos(-dot(v01, v12) / (L01 * L12));
    auto A3 = M_PI - A1 - A2;

    return {A1, A2, A3};
}

inline double vec_angle(const Vec3& v1, const Vec3& v2) {
    auto v1L = length(v1);
    auto v2L = length(v2);
    auto v1d2 = dot(v1, v2);
    auto arg = v1d2 / (v1L * v2L);
    if (arg < -1) {
        arg = -1;
    } else if (arg > 1) {
        arg = 1;
    }
    return acos(arg);
}

inline Vec3 get_edge_lens(const Tensor3& tri) {
    Vec3 out;
    for (int d = 0; d < 3; d++) {
        out[d] = 0.0;
        for (int c = 0; c < 3; c++) {
            auto delta = tri[(d + 1) % 3][c] - tri[d][c];
            out[d] += delta * delta;
        }
    }
    return out;
}

inline int get_longest_edge(const Vec3& lens) {
    if (lens[0] >= lens[1] && lens[0] >= lens[2]) {
        return 0;
    } else if (lens[1] >= lens[0] && lens[1] >= lens[2]) {
        return 1;
    } else {// if (lens[2] >= lens[0] && lens[2] >= lens[1]) {
        return 2;
    }
}

