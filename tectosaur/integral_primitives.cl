<%!
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

<%def name="geometry_fncs()">
void vec_cross(Real x[3], Real y[3], Real out[3]) {
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
}

void sub(Real x[3], Real y[3], Real out[3]) {
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
}

void get_unscaled_normal(Real tri[3][3], Real out[3]) {
    Real s20[3];
    Real s21[3];
    sub(tri[2], tri[0], s20);
    sub(tri[2], tri[1], s21);
    vec_cross(s20, s21, out);
}

Real magnitude(Real v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
</%def>

<%def name="get_triangle(name, tris, index)">
Real ${name}[3][3];
for (int c = 0; c < 3; c++) {
    for (int d = 0; d < 3; d++) {
        ${name}[c][d] = pts[3 * ${tris}[3 * ${index} + c] + d];
    }
}
</%def>

<%def name="tri_info(prefix,normal_prefix)">
Real ${prefix}_unscaled_normal[3];
get_unscaled_normal(${prefix}_tri, ${prefix}_unscaled_normal);
Real ${prefix}_normal_length = magnitude(${prefix}_unscaled_normal);
Real ${prefix}_jacobian = ${prefix}_normal_length;
% for dim in range(3):
Real ${normal_prefix}${dn(dim)} = 
    ${prefix}_unscaled_normal[${dim}] / ${prefix}_normal_length;
% endfor
</%def>

<%def name="basis(prefix)">
Real ${prefix}b0 = 1 - ${prefix}xhat - ${prefix}yhat;
Real ${prefix}b1 = ${prefix}xhat;
Real ${prefix}b2 = ${prefix}yhat;
</%def>

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name,ndims)">
% for dim in range(ndims):
Real ${pt_pfx}${dn(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dn(dim)} += ${basis_pfx}b${basis} * ${tri_name(basis,dim)};
% endfor
% endfor
</%def>

<%def name="temp_result_idx(d_obs, d_src, b_obs, b_src)">
${b_obs} * 27 + ${d_obs} * 9 + ${b_src} * 3 + ${d_src}
</%def>

<%def name="call_vector_code(K)">
% if K.vector_code is None:
    ${K.tensor_code}
    % for d1 in range(K.tensor_dim):
        % for d2 in range(K.tensor_dim):
            sum${dn(d1)} += K${d1}${d2} * in${dn(d2)};
        % endfor
    % endfor
% else:
    ${K.vector_code}
% endif
</%def>

