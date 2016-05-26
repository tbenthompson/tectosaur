<%namespace file="/coincident_helpers.cpp" name="helpers"/>

<%
dim_name = helpers.attr.dim_name
%>

#include <stdio.h>

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

__device__
void coincident_wrapped(float* result, int n_quad_pts, float* quad_pts_d,
    float* quad_wts_d, float* pts_d, int* tris_d, float G, float nu)
{
    auto res_index = [=] (int, int bobs, int bsrc, int d1, int d2) {
        return bobs * 27 + bsrc * 9 + d1 * 3 + d2;
    };

    auto tri_index = [=] (int i, int d) {
        return i * 3 + d;
    };

    auto pts_index = tri_index;

    auto quad_index = [=] (int i, int d1) {
        return i * 5 + d1;
    };

    const int it = blockIdx.x * ${block[0]} + threadIdx.x;

    Tri tri;
    for (int c = 0; c < 3; c++) {
        for (int d = 0; d < 3; d++) {
            tri[c][d] = pts_d[pts_index(tris_d[tri_index(it, c)], d)];
        }
    }

    auto unscaled_normal = get_unscaled_normal(tri);
    float normal_length = magnitude(unscaled_normal);
    float jacobian = normal_length;

    % for dim in range(3):
    float n${dim_name(dim)} = unscaled_normal[${dim}] / normal_length;
    float l${dim_name(dim)} = n${dim_name(dim)};
    % endfor


    float result_d[81];

    for (int iresult = 0; iresult < 81; iresult++) {
        result_d[iresult] = 0;
    }

    for (int iq = 0; iq < n_quad_pts; iq++) {
        float eps = quad_pts_d[quad_index(iq, 4)];
        float obsxhat = quad_pts_d[quad_index(iq, 0)];
        float obsyhat = quad_pts_d[quad_index(iq, 1)];
        float srcxhat = quad_pts_d[quad_index(iq, 2)];
        float srcyhat = quad_pts_d[quad_index(iq, 3)];
        float quadw = quad_wts_d[iq];

        ${helpers.basis("obs")}
        ${helpers.pts_from_basis("x", "obs")}
        % for dim in range(3):
        x${dim_name(dim)} -= eps * unscaled_normal[${dim}];
        % endfor

        ${helpers.basis("src")}
        ${helpers.pts_from_basis("y", "src")}

        ${helpers.call_kernel('H')}
    }
    
    ${helpers.enforce_symmetry('H')}

    for (int iresult = 0; iresult < 81; iresult++) {
        result[it * 81 + iresult] = result_d[iresult];
    }
}

extern "C" {
    __global__ 
    void coincident(float* result, int n_quad_pts, float* quad_pts,
        float* quad_wts, float* pts, int* tris, float G, float nu) 
    {
        coincident_wrapped(result, n_quad_pts, quad_pts, quad_wts, pts, tris, G, nu);
    }
}
