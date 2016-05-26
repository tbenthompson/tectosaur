<%!
import tectosaur.elastic as elastic
kernel_names = ['U', 'T', 'A', 'H']
kernels = dict()
for k_name in kernel_names:
    kernels[k_name] = elastic.get_kernel(getattr(elastic, k_name))
def dim_name(dim):
    return ['x', 'y', 'z'][dim]
%>

<%def name="pts_from_basis(pt_pfx,basis_pfx)">
% for dim in range(3):
Real ${pt_pfx}${dim_name(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dim_name(dim)} += ${basis_pfx}b${basis} * tri[${basis}][${dim}];
% endfor
% endfor
</%def>

<%def name="basis(prefix)">
auto ${prefix}b0 = 1 - ${prefix}xhat - ${prefix}yhat;
auto ${prefix}b1 = ${prefix}xhat;
auto ${prefix}b2 = ${prefix}yhat;
</%def>

<%def name="call_kernel(k_name)">
% for d_obs in range(3):
<% 
max_d_src = 3
if kernels[k_name]['symmetric']:
    max_d_src = d_obs + 1
%>
% for d_src in range(max_d_src):
{
    Real kernel_val = jacobian * quadw * ${kernels[k_name]['expr'][d_obs][d_src]};
    % for b_obs in range(3):
    % for b_src in range(3):
    result_d[res_index(it, ${b_obs}, ${b_src}, ${d_obs}, ${d_src})] += 
        obsb${b_obs} * srcb${b_src} * kernel_val;
    % endfor
    % endfor
}
% endfor
% endfor
</%def>

<%def name="enforce_symmetry(k_name)">
% if kernels[k_name]['symmetric']:
for (size_t b1 = 0; b1 < 3; b1++) {
    for (size_t b2 = 0; b2 < 3; b2++) {
        for (size_t d1 = 0; d1 < 3; d1++) {
            for (size_t d2 = 0; d2 < d1; d2++) {
                result_d[res_index(it, b1, b2, d2, d1)] =
                    result_d[res_index(it, b1, b2, d1, d2)];
            }
        }
    }
}
% endif
</%def>
