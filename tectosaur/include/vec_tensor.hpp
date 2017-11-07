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
