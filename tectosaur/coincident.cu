<%namespace file="/coincident_helpers.cpp" name="helpers"/>

<%
dim_name = helpers.attr.dim_name
%>

#include <stdio.h>
#include "cuda_geometry.hpp"

<%def name="coincident_gpu(k_name)">
__device__
void coincident_wrapped${k_name}(float* result, int n_quad_pts, float* quad_pts_d,
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

    const int it = blockIdx.x * blockDim.x + threadIdx.x;

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
    n${dim_name(dim)} = l${dim_name(dim)};
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
        ${helpers.pts_from_basis("x", "obs", "tri")}
        % for dim in range(3):
        x${dim_name(dim)} -= eps * unscaled_normal[${dim}];
        % endfor

        ${helpers.basis("src")}
        ${helpers.pts_from_basis("y", "src", "tri")}

        ${helpers.call_kernel(k_name)}
    }
    
    ${helpers.enforce_symmetry(k_name)}

    for (int iresult = 0; iresult < 81; iresult++) {
        result[it * 81 + iresult] = result_d[iresult];
    }
}

extern "C" {
    __global__ 
    void coincident${k_name}(float* result, int n_quad_pts, float* quad_pts,
        float* quad_wts, float* pts, int* tris, float G, float nu) 
    {
        coincident_wrapped${k_name}(
            result, n_quad_pts, quad_pts, quad_wts, pts, tris, G, nu
        );
    }
}
</%def>

% for k_name in helpers.attr.kernel_names:
${coincident_gpu(k_name)}
% endfor
