<%!
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

<%namespace name="kernels" file="kernels/kernels.cl"/>

<%def name="geometry_fncs()">

CONSTANT Real basis_gradient[3][2] = {{-1.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}};

WITHIN_KERNEL void vec_cross(Real x[3], Real y[3], Real out[3]) {
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
}

WITHIN_KERNEL void sub(Real x[3], Real y[3], Real out[3]) {
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
}

WITHIN_KERNEL void get_unscaled_normal(Real tri[3][3], Real out[3]) {
    Real s20[3];
    Real s21[3];
    sub(tri[2], tri[0], s20);
    sub(tri[2], tri[1], s21);
    vec_cross(s20, s21, out);
}

WITHIN_KERNEL Real magnitude(Real v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

WITHIN_KERNEL int positive_mod(int i, int n) {
        return (i % n + n) % n;
}
</%def>

<%def name="decl_tri_info(name, need_normal, need_surf_curl)">
Real ${name}_tri[3][3];
Real ${name}_unscaled_normal[3];
Real ${name}_normal_length;
Real ${name}_jacobian;
% if need_normal:
    % for dim in range(3):
        Real n${name}${dn(dim)};
    % endfor
% endif
% if need_surf_curl:
    Real b${name}_surf_curl[3][3];
% endif
</%def>

<%def name="tri_info(name, tris, need_normal, need_surf_curl)">
for (int c = 0; c < 3; c++) {
    int pt_idx = ${tris}[
        3 * ${name}_tri_idx + positive_mod(c + ${name}_tri_rot_clicks, 3)
    ];
    for (int d = 0; d < 3; d++) {
        ${name}_tri[c][d] = pts[pt_idx * 3 + d];
    }
}
get_unscaled_normal(${name}_tri, ${name}_unscaled_normal);
${name}_normal_length = magnitude(${name}_unscaled_normal);
${name}_jacobian = ${name}_normal_length;
% if need_normal:
    % for dim in range(3):
    n${name}${dn(dim)} = 
        ${name}_unscaled_normal[${dim}] / ${name}_normal_length;
    % endfor
% endif
% if need_surf_curl:
    ${surf_curl_basis(name)}
% endif
</%def>

<%def name="surf_curl_basis(name)">
// The output is indexed as:
// b{name}_surf_curl[basis_idx][curl_component]
{
    Real g1[3];
    Real g2[3];
    sub(${name}_tri[1], ${name}_tri[0], g1);
    sub(${name}_tri[2], ${name}_tri[0], g2);
    for (int basis_idx = 0; basis_idx < 3; basis_idx++) {
        for (int s = 0; s < 3; s++) {
            b${name}_surf_curl[basis_idx][s] = (
                + basis_gradient[basis_idx][0] * g2[s] 
                - basis_gradient[basis_idx][1] * g1[s]
            ) / ${name}_jacobian;
        }
    }
}
</%def>

<%def name="basis(prefix)">
Real ${prefix}b[3] = {
    1 - ${prefix}xhat - ${prefix}yhat, ${prefix}xhat, ${prefix}yhat
};
</%def>


<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name,ndims)">
% for dim in range(ndims):
Real ${pt_pfx}${dn(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dn(dim)} += ${basis_pfx}b[${basis}] * ${tri_name(basis,dim)};
% endfor
% endfor
</%def>

<%def name="temp_result_idx(d_obs, d_src, b_obs, b_src)">
${b_obs} * 27 + ${d_obs} * 9 + ${b_src} * 3 + ${d_src}
</%def>

<%def name="call_vector_code(K)">
% if hasattr(kernels, K.name + '_vector'):
    ${getattr(kernels, K.name + '_vector')()}
% elif K.vector_code is None:
    Real Karr[${K.tensor_dim} * ${K.tensor_dim}];
    ${K.tensor_code}
    % for d1 in range(K.tensor_dim):
        % for d2 in range(K.tensor_dim):
            sum${dn(d1)} += Karr[${d1} * ${K.tensor_dim} + ${d2}] * in${dn(d2)};
        % endfor
    % endfor
% else:
    ${K.vector_code}
% endif
</%def>

<%def name="call_tensor_code(K)">
% if hasattr(kernels, K.name + '_tensor'):
    ${getattr(kernels, K.name + '_tensor')()}
% else:
    Real Karr[9];
    ${K.tensor_code}
    obsb[0] *= quadw;
    obsb[1] *= quadw;
    obsb[2] *= quadw;

    int idx = 0;
    for (int b_obs = 0; b_obs < 3; b_obs++) {
    for (int d_obs = 0; d_obs < 3; d_obs++) {
    for (int b_src = 0; b_src < 3; b_src++) {
    for (int d_src = 0; d_src < 3; d_src++, idx++) {
        Real val = obsb[b_obs] * srcb[b_src] * Karr[d_obs * 3 + d_src];
        Real y = val - kahanC[idx];
        Real t = result_temp[idx] + y;
        kahanC[idx] = (t - result_temp[idx]) - y;
        result_temp[idx] = t;
    }
    }
    }
    }
% endif
</%def>

