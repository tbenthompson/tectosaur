#include <array>

using Vec3 = std::array<double,3>;
using Tensor3 = std::array<Vec3,3>;

Tensor3 transpose(const Tensor3& A) {
    Tensor3 out;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1][d2] = A[d2][d1];
        }
    }
    return out;
}

Tensor3 mult(const Tensor3& A, double factor) {
    Tensor3 out;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1][d2] = A[d1][d2] * factor;
        }
    }
    return out;
}

Tensor3 mult(const Tensor3& A, const Tensor3& B) {
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

Tensor3 outer(const Vec3& A, const Vec3& B) {
    Tensor3 out;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1][d2] = A[d1] * B[d2];
        }
    }
    return out;
}

Tensor3 rotation_matrix(const Vec3& axis, double theta) {
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

Vec3 sub(const Vec3& A, const Vec3& B) {
    return {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
}

Vec3 add(const Vec3& A, const Vec3& B) {
    return {A[0] + B[0], A[1] + B[1], A[2] + B[2]};
}

Vec3 div(const Vec3& A, double f) {
    double multfactor = 1.0 / f;
    return {A[0] * multfactor, A[1] * multfactor, A[2] * multfactor};
}

double length(const Vec3& A) {
    return std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
}
