<%
from tectosaur.integral_utils import dn, kernel_names
%>

#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define Real ${float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

<%def name="interior_integrals(k_name)">
__kernel
void interior_integrals${k_name}(__global Real* result, 
    int n_quad_pts, __global Real* quad_pts, __global Real* quad_wts,
    __global Real* obs_pts, __global Real* obs_ns,
    __global Real* pts, int n_src_tris, __global int* src_tris,
    __global Real* input, Real G, Real nu)
{
    const int i = get_global_id(0);

    % for d in range(3):
    Real obsp${dn(d)} = obs_pts[i * 3 + ${d}];
    Real n${dn(d)} = obs_ns[i * 3 + ${d}];
    % endfor

    Real sumx = 0.0;
    Real sumy = 0.0;
    Real sumz = 0.0;

    ${prim.constants()}

    for (int j = 0; j < n_src_tris; j++) {
        ${prim.get_triangle("src_tri", "src_tris", "j")}
        ${prim.tri_info("src", "l")}
        Real corner_vals[3][3];
        for (int c = 0; c < 3; c++) {
            for (int d = 0; d < 3; d++) {
                corner_vals[c][d] = input[j * 9 + c * 3 + d];
            }
        }

        for (int iq = 0; iq < n_quad_pts; iq++) {
            Real srcxhat = quad_pts[iq * 2 + 2];
            Real srcyhat = quad_pts[iq * 2 + 3];
            Real quadw = quad_wts[iq];
            ${prim.basis("src")}
            ${prim.pts_from_basis(
                "y", "src",
                lambda b, d: "src_tri[" + str(b) + "][" + str(d) + "]", 3
            )}
            ${prim.pts_from_basis(
                "S", "src",
                lambda b, d: "corner_vals[" + str(b) + "][" + str(d) + "]", 3
            )}

            Real Dx = yx - obspx;
            Real Dy = yy - obspy; 
            Real Dz = yz - obspz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            ${prim.vector_kernels(k_name)}
        }
    }

    % for d in range(3):
    result[i * 3 + ${d}] = sum${dn(d)};
    % endfor
}
</%def>

${prim.geometry_fncs()}

% for k_name in kernel_names:
${interior_integrals(k_name)}
% endfor
