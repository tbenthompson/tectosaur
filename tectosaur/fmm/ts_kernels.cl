<%!
def dn(dim):
    return ['x', 'y', 'z'][dim]
from tectosaur.kernels import kernels
e = [[[int((i - j) * (j - k) * (k - i) / 2) for k in range(3)]
    for j in range(3)] for i in range(3)]
%>
<%
    multipole_dim = K.multipole_dim
%>
${cluda_preamble}

<%def name="p2m_core(n, m, real, imag)">
    %if K.name == "elasticU3":
        % for dobs in range(3):
            sumreal[${n}][${m}][${dobs}] += ${real} * invals[${dobs}];
            sumimag[${n}][${m}][${dobs}] += ${imag} * invals[${dobs}];
        % endfor
    % elif K.name == "elasticRT3":
        % for dobs in range(3):
            % for dsrc in range(3):
                % for j in range(3):
                    sumreal[${n}][${m}][${dobs}] += (
                        ${e[dobs][dsrc][j]} * ${real} * src_surf_curl[${dsrc}][${j}]
                    );
                    sumimag[${n}][${m}][${dobs}] += (
                        ${e[dobs][dsrc][j]} * ${imag} * src_surf_curl[${dsrc}][${j}]
                    );
                % endfor
            % endfor
        % endfor
    % elif K.name == "elasticRA3":
        % for dsrc in range(3):
            sumreal[${n}][${m}][${dsrc}] += ${real} * invals[${dsrc}];
            sumimag[${n}][${m}][${dsrc}] += ${imag} * invals[${dsrc}];
        % endfor
    % elif K.name == "elasticRH3":
        % for dsrc in range(3):
            sumreal[${n}][${m}][0] += ${real} * src_surf_curl[${dsrc}][${dsrc}];
            sumimag[${n}][${m}][0] += ${imag} * src_surf_curl[${dsrc}][${dsrc}];
        % endfor
        % for dobs in range(3):
            % for j in range(3):
                sumreal[${n}][${m}][1 + ${dobs * 3 + j}] += (
                    ${real} * src_surf_curl[${dobs}][${j}]
                );
                sumimag[${n}][${m}][1 + ${dobs * 3 + j}] += (
                    ${imag} * src_surf_curl[${dobs}][${j}]
                );
            % endfor
        % endfor
    % endif
</%def>

<%def name="p2md_core(dsn, dsm, d_idx, real, imag)">
{
    % if K.name == "elasticU3":
        sumreal[${dsn}][${dsm}][3] += ${real} * invals[${d_idx}];
        sumimag[${dsn}][${dsm}][3] += ${imag} * invals[${d_idx}];
        % for dobs in range(3):
            sumreal[${dsn}][${dsm}][4 + ${dobs}] += (
                D${dn(dobs)} * ${real} * invals[${d_idx}]
            );
            sumimag[${dsn}][${dsm}][4 + ${dobs}] += (
                D${dn(dobs)} * ${imag} * invals[${d_idx}]
            );
        % endfor
    % elif K.name == "elasticRT3":
        % for j in range(3):
            % for dsrc in range(3):
                sumreal[${dsn}][${dsm}][3] += (
                    ${e[j][d_idx][dsrc]} * ${real} * src_surf_curl[${dsrc}][${j}]
                );
                sumimag[${dsn}][${dsm}][3] += (
                    ${e[j][d_idx][dsrc]} * ${imag} * src_surf_curl[${dsrc}][${j}]
                );
            % endfor
        % endfor
        % for dobs in range(3):
            % for j in range(3):
                % for dsrc in range(3):
                    sumreal[${dsn}][${dsm}][4 + ${dobs}] += (
                        ${e[j][d_idx][dsrc]} * D${dn(dobs)} * ${real} 
                        * src_surf_curl[${dsrc}][${j}]
                    );
                    sumimag[${dsn}][${dsm}][4 + ${dobs}] += (
                        ${e[j][d_idx][dsrc]} * D${dn(dobs)} * ${imag} 
                        * src_surf_curl[${dsrc}][${j}]
                    );
                % endfor
            % endfor
        % endfor
    % elif K.name == "elasticRA3":
        sumreal[${dsn}][${dsm}][3] += ${real} * invals[${d_idx}];
        sumimag[${dsn}][${dsm}][3] += ${imag} * invals[${d_idx}];
        % for p in range(3):
            sumreal[${dsn}][${dsm}][4 + ${p}] += (
                D${dn(p)} * ${real} * invals[${d_idx}]
            );
            sumimag[${dsn}][${dsm}][4 + ${p}] += (
                D${dn(p)} * ${imag} * invals[${d_idx}]
            );
        % endfor
    % elif K.name == "elasticRH3":
        % for j in range(3):
            sumreal[${dsn}][${dsm}][10 + ${j}] += ${real} * src_surf_curl[${d_idx}][${j}];
            sumimag[${dsn}][${dsm}][10 + ${j}] += ${imag} * src_surf_curl[${d_idx}][${j}];
            % for dobs in range(3):
                sumreal[${dsn}][${dsm}][13 + ${dobs * 3 + j}] += D${dn(dobs)} * ${real} * src_surf_curl[${d_idx}][${j}];
                sumimag[${dsn}][${dsm}][13 + ${dobs * 3 + j}] += D${dn(dobs)} * ${imag} * src_surf_curl[${d_idx}][${j}];
            % endfor
        % endfor
    % endif
}
</%def>

<%def name="m2m_core()">
% if K.name == "elasticU3" or K.name == "elasticRT3":
    %for d in range(3):
    {
        ${get_child_multipoles(d)}
        sumreal[ni][mi][${d}] += realval;
        sumimag[ni][mi][${d}] += imagval;
    }
    % endfor

    ${get_child_multipoles(3)}
    sumreal[ni][mi][3] += realval;
    sumimag[ni][mi][3] += imagval;
    %for d in range(3):
    {
        sumreal[ni][mi][${4 + d}] += D${dn(d)} * realval;
        sumimag[ni][mi][${4 + d}] += D${dn(d)} * imagval;
    }
    % endfor

    %for d in range(3):
    {
        ${get_child_multipoles(4 + d)}
        sumreal[ni][mi][${4 + d}] += realval;
        sumimag[ni][mi][${4 + d}] += imagval;
    }
    % endfor
% elif K.name == "elasticRA3":
% elif K.name == "elasticRH3":
% endif
</%def>

<%def name="m2p_core(n, real, imag)">
{
    % if K.name == "elasticU3":
        Real mult = 1.0;
        if (mi > 0) {
            mult = 2.0;
        }

        {
            Real C = mult * CsU0;
            Real real_val = C * (${real});
            Real imag_val = C * (${imag});
            for (int d = 0; d < 3; d++) {
                int idx =
                    (${n}) * ${multipole_dim * 2 * (order + 1)}
                    + mi * ${multipole_dim} * 2
                    + d * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                sum[d] += Rr * real_val + Ri * imag_val;
            }
        }

        {
            Real C = mult * CsU1;
            Real real_val = C * (${real}); 
            Real imag_val = C * (${imag}); 

            int idx = 
                (${n}) * ${multipole_dim * 2 * (order + 1)} 
                + mi * ${multipole_dim * 2} 
                + 3 * 2;
            Real Rr = sh_multipoles[idx];
            Real Ri = sh_multipoles[idx + 1];
            Real Aval = Rr * real_val + Ri * imag_val;
            % for d1 in range(3):
            {
                int idx = 
                    (${n}) * ${multipole_dim * 2 * (order + 1)} 
                    + mi * ${multipole_dim * 2} 
                    + (4 + ${d1}) * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                Real Bval = Rr * real_val + Ri * imag_val;

                sum[${d1}] += D${dn(d1)} * Aval - Bval;
            }
            % endfor
        }
    % elif K.name == "elasticRT3":
        Real mult = 1.0;
        if (mi > 0) {
            mult = 2.0;
        }

        {
            Real C = mult * CsRT1;
            Real real_val = C * (${real});
            Real imag_val = C * (${imag});
            for (int d = 0; d < 3; d++) {
                int idx =
                    (${n}) * ${multipole_dim * 2 * (order + 1)}
                    + mi * ${multipole_dim} * 2
                    + d * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                sum[d] += Rr * real_val + Ri * imag_val;
            }
        }

        
        {
            Real C = mult * CsRT0;
            Real real_val = C * (${real}); 
            Real imag_val = C * (${imag}); 

            int idx = 
                (${n}) * ${multipole_dim * 2 * (order + 1)} 
                + mi * ${multipole_dim * 2} 
                + 3 * 2;
            Real Rr = sh_multipoles[idx];
            Real Ri = sh_multipoles[idx + 1];
            Real Aval = Rr * real_val + Ri * imag_val;
            % for d1 in range(3):
            {
                int idx = 
                    (${n}) * ${multipole_dim * 2 * (order + 1)} 
                    + mi * ${multipole_dim * 2} 
                    + (4 + ${d1}) * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                Real Bval = Rr * real_val + Ri * imag_val;

                sum[${d1}] += D${dn(d1)} * Aval - Bval;
            }
            % endfor
        }
    % elif K.name == "elasticRA3":
        Real mult = 1.0;
        if (mi > 0) {
            mult = 2.0;
        }

        {
            Real C = mult * CsRT1;
            Real real_val = C * (${real});
            Real imag_val = C * (${imag});
            % for dsrc in range(3):
            {
                int idx =
                    (${n}) * ${multipole_dim * 2 * (order + 1)}
                    + mi * ${multipole_dim} * 2
                    + ${dsrc} * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                Real SR = Rr * real_val + Ri * imag_val;
                % for dobs in range(3):
                    % for j in range(3):
                        for (int bobs = 0; bobs < 3; bobs++) {
                            Real term = (
                                ${e[dsrc][dobs][j]}
                                * bobs_surf_curl[bobs][${j}] 
                                * SR
                            );
                            basissum[bobs][${dobs}] += term;
                        }
                    % endfor
                % endfor
            }
            % endfor
        }

        {
            Real C = mult * CsRT0;
            Real real_val = C * (${real}); 
            Real imag_val = C * (${imag}); 

            int idx = 
                (${n}) * ${multipole_dim * 2 * (order + 1)} 
                + mi * ${multipole_dim * 2} 
                + 3 * 2;
            Real Rr = sh_multipoles[idx];
            Real Ri = sh_multipoles[idx + 1];
            Real Aval = Rr * real_val + Ri * imag_val;
            % for p in range(3):
            {
                int idx = 
                    (${n}) * ${multipole_dim * 2 * (order + 1)} 
                    + mi * ${multipole_dim * 2} 
                    + (4 + ${p}) * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                Real Bval = Rr * real_val + Ri * imag_val;
                % for dobs in range(3):
                % for j in range(3):
                    for (int bobs = 0; bobs < 3; bobs++) {
                        basissum[bobs][${dobs}] += (
                            ${e[j][p][dobs]} 
                            * (D${dn(p)} * Aval - Bval) 
                            * bobs_surf_curl[bobs][${j}]
                        );
                    }
                % endfor
                % endfor
            }
            % endfor
        }
    % elif K.name == "elasticRH3":
        Real mult = 1.0;
        if (mi > 0) {
            mult = 2.0;
        }
        {
            Real C = mult * CsRH0;
            Real real_val = C * (${real});
            Real imag_val = C * (${imag});
            {
                int idx =
                    (${n}) * ${multipole_dim * 2 * (order + 1)}
                    + mi * ${multipole_dim} * 2
                    + 0 * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                Real SR = CsRH1 * (Rr * real_val + Ri * imag_val);
                % for dobs in range(3):
                    for (int bobs = 0; bobs < 3; bobs++) {
                        basissum[bobs][${dobs}] += bobs_surf_curl[bobs][${dobs}] * SR;
                    }
                % endfor
            }

            % for dobs in range(3):
            % for j in range(3):
            {
                int idx =
                    (${n}) * ${multipole_dim * 2 * (order + 1)}
                    + mi * ${multipole_dim} * 2
                    + (1 + ${dobs * 3 + j}) * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                Real SR = Rr * real_val + Ri * imag_val;
                for (int bobs = 0; bobs < 3; bobs++) {
                    basissum[bobs][${dobs}] -= 2 * bobs_surf_curl[bobs][${j}] * SR;
                    basissum[bobs][${j}] += CsRH2 * bobs_surf_curl[bobs][${dobs}] * SR;
                }
            }
            % endfor
            % endfor

            % for j in range(3):
            {
                int idx =
                    (${n}) * ${multipole_dim * 2 * (order + 1)}
                    + mi * ${multipole_dim} * 2
                    + (10 + ${j}) * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                Real SR = Rr * real_val + Ri * imag_val;
                % for dobs in range(3):
                for (int bobs = 0; bobs < 3; bobs++) {
                    basissum[bobs][${dobs}] -= 2 * D${dn(dobs)} * bobs_surf_curl[bobs][${j}] * SR;
                }
                % endfor
            }
            % endfor

            % for dobs in range(3):
            % for j in range(3):
            {
                int idx =
                    (${n}) * ${multipole_dim * 2 * (order + 1)}
                    + mi * ${multipole_dim} * 2
                    + (13 + ${dobs * 3 + j}) * 2;
                Real Rr = sh_multipoles[idx];
                Real Ri = sh_multipoles[idx + 1];
                Real SR = Rr * real_val + Ri * imag_val;
                for (int bobs = 0; bobs < 3; bobs++) {
                    basissum[bobs][${dobs}] += 2 * bobs_surf_curl[bobs][${j}] * SR;
                }
            }
            % endfor
            % endfor
        }

    % endif
}
</%def>

#define Real ${gpu_float_type}

<%namespace name="prim" file="../integral_primitives.cl"/>

${prim.geometry_fncs()}

CONSTANT Real quad_pts[${quad_pts.size}] = {${str(quad_pts.flatten().tolist())[1:-1]}};
CONSTANT Real quad_wts[${quad_wts.size}] = {${str(quad_wts.flatten().tolist())[1:-1]}};

<%def name="multipole_idx(node_idx, n_idx, m_idx, d_idx)">
    (${node_idx} * ${multipole_dim * 2 * (order + 1) * (order + 1)}
    + ${n_idx} * ${multipole_dim * 2 * (order + 1)}
    + ${m_idx} * ${multipole_dim * 2}
    + ${d_idx} * 2)
</%def>

KERNEL void p2p(
    GLOBAL_MEM Real* out,
    GLOBAL_MEM Real* inarr,
    int n_obs_tris,
    GLOBAL_MEM Real* params,
    GLOBAL_MEM int* obs_tri_block_idxs,
    GLOBAL_MEM int* obs_src_starts,
    GLOBAL_MEM int* src_n_idxs,
    GLOBAL_MEM int* obs_n_start,
    GLOBAL_MEM int* obs_n_end,
    GLOBAL_MEM Real* obs_pts,
    GLOBAL_MEM int* obs_tris,
    GLOBAL_MEM int* src_n_start,
    GLOBAL_MEM int* src_n_end,
    GLOBAL_MEM Real* src_pts,
    GLOBAL_MEM int* src_tris)
{
    const int obs_tri_idx = get_global_id(0); 
    if (obs_tri_idx >= n_obs_tris) {
        return;
    }
    const int block_idx = obs_tri_block_idxs[obs_tri_idx];
    const int this_obs_n_idx = block_idx;
    const int this_obs_src_start = obs_src_starts[block_idx];
    const int this_obs_src_end = obs_src_starts[block_idx + 1];

    ${K.constants_code}

    const int obs_tri_rot_clicks = 0;
    ${prim.decl_tri_info("obs", K.needs_obsn, K.surf_curl_obs)}
    ${prim.tri_info("obs", "obs_pts", "obs_tris", K.needs_obsn, K.surf_curl_obs)}

    Real sum[9];
    for (int d = 0; d < 9; d++) {
        sum[d] = 0.0;
    }

    for (int src_block_idx = this_obs_src_start;
         src_block_idx < this_obs_src_end;
         src_block_idx++) 
    {
        const int this_src_n_idx = src_n_idxs[src_block_idx];
        for (int src_tri_idx = src_n_start[this_src_n_idx];
                src_tri_idx < src_n_end[this_src_n_idx];
                src_tri_idx++) 
        {
            const int src_tri_rot_clicks = 0;
            ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
            ${prim.tri_info("src", "src_pts", "src_tris", K.needs_srcn, K.surf_curl_src)}

            Real in[9];
            for (int k = 0; k < 9; k++) {
                in[k] = inarr[src_tri_idx * 9 + k];
            }

            % for iq1 in range(quad_wts.shape[0]):
            {
                Real obsxhat = ${quad_pts[iq1,0]};
                Real obsyhat = ${quad_pts[iq1,1]};
                % for iq2 in range(quad_wts.shape[0]):
                {
                    Real srcxhat = ${quad_pts[iq2,0]};
                    Real srcyhat = ${quad_pts[iq2,1]};
                    Real quadw = ${quad_wts[iq1] * quad_wts[iq2]};
                    % for which, ptname in [("obs", "x"), ("src", "y")]:
                        ${prim.basis(which)}
                        ${prim.pts_from_basis(
                            ptname, which,
                            lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
                        )}
                    % endfor

                    Real Dx = yx - xx;
                    Real Dy = yy - xy; 
                    Real Dz = yz - xz;
                    Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

                    % if iq1 == iq2:
                        if (r2 == 0.0) {
                            continue;
                        }
                    % endif

                    Real factor = obs_jacobian * src_jacobian * quadw;
                    % for d in range(3):
                        Real sum${dn(d)} = 0.0;
                        Real in${dn(d)} = 0.0;
                        for (int b_src = 0; b_src < 3; b_src++) {
                            in${dn(d)} += in[b_src * 3 + ${d}] * srcb[b_src];
                        }
                    % endfor

                    ${prim.call_vector_code(K)}

                    for (int b_obs = 0; b_obs < 3; b_obs++) {
                        % for d_obs in range(3):
                        sum[b_obs * 3 + ${d_obs}] += factor * obsb[b_obs] * sum${dn(d_obs)};
                        % endfor
                    }
                }
                % endfor
            }
            % endfor
        }
    }
    for (int k = 0; k < 9; k++) {
        out[obs_tri_idx * 9 + k] = sum[k];
    }
}

<%def name="multipole_sum(n, m, real, imag)">
    ${p2m_core(n, m, real, imag)}
    ${out_sum_dR(n, m, real, imag)}
    if (mi == 1) {
        ${out_sum_dR_plus1(
            "(" + n + ")", "(-1)", "(-" + real + ")", "(" + imag + ")"
        )}
    }
</%def>

<%def name="out_sum_dR(n, m, real, imag)">
    ${out_sum_dR_minus1(n, m, real, imag)}
    ${out_sum_dR_plus1(n, m, real, imag)}
    ${out_sum_dR_0(n, m, real, imag)}
</%def>

<%def name="out_sum_dR_minus1(n, m, real, imag)">
{
    int dRn = (${n}) + 1;
    int dRmm1 = (${m}) - 1;
    if (0 <= dRn && dRmm1 >= 0 && dRn <= ${order}) {
        Real dRr1 = -0.5 * (${real});    
        Real dRi1 = -0.5 * (${imag});        
        ${p2md_core("dRn", "dRmm1", 0, "dRr1", "dRi1")}
        Real dRr2 = -0.5 * (${imag});    
        Real dRi2 = 0.5 * (${real});        
        ${p2md_core("dRn", "dRmm1", 1, "dRr2", "dRi2")}
    }
}
</%def>

<%def name="out_sum_dR_plus1(n, m, real, imag)">
{
    int dRn = (${n}) + 1;
    int dRmp1 = (${m}) + 1;
    if (0 <= dRn && dRmp1 <= dRn && dRn <= ${order}) {
        Real dRr1 = 0.5 * (${real});
        Real dRi1 = 0.5 * (${imag});
        ${p2md_core("dRn", "dRmp1", 0, "dRr1", "dRi1")}
        Real dRr2 = -0.5 * (${imag});
        Real dRi2 =  0.5 * (${real});
        ${p2md_core("dRn", "dRmp1", 1, "dRr2", "dRi2")}
    }
}
</%def>

<%def name="out_sum_dR_0(n, m, real, imag)">
{
    int dRn = (${n}) + 1;
    if (0 <= dRn && ${m} <= dRn && dRn <= ${order}) {
        Real dRr3 = (${real});
        Real dRi3 = (${imag});
        ${p2md_core("dRn", m, 2, "dRr3", "dRi3")}
    }
}
</%def>

<%def name="finish_multipole_sum()">
    for (int i = 0; i < ${order + 1}; i++) {
        for (int j = 0; j < ${order + 1}; j++) {
            for (int d = 0; d < ${multipole_dim}; d++) {
                int idx = ${multipole_idx("this_obs_n_idx", "i", "j", "d")};
                multipoles[idx] = sumreal[i][j][d];
                multipoles[idx + 1] = sumimag[i][j][d];
            }
        }
    }
</%def>

KERNEL void p2m(
    GLOBAL_MEM Real* multipoles,
    GLOBAL_MEM Real* src_in,
    int n_blocks,
    GLOBAL_MEM int* n_idxs,
    GLOBAL_MEM Real* n_centers,
    GLOBAL_MEM int* n_starts,
    GLOBAL_MEM int* n_ends,
    GLOBAL_MEM Real* src_pts,
    GLOBAL_MEM int* src_tris)
{
    const int global_idx = get_global_id(0); 
    const int block_idx = global_idx;
    if (block_idx >= n_blocks) {
        return;
    }
    const int this_obs_n_idx = n_idxs[block_idx];

    Real xx = n_centers[this_obs_n_idx * 3 + 0];
    Real xy = n_centers[this_obs_n_idx * 3 + 1];
    Real xz = n_centers[this_obs_n_idx * 3 + 2];

    Real sumreal[${order + 1}][${order + 1}][${multipole_dim}];
    Real sumimag[${order + 1}][${order + 1}][${multipole_dim}];

    for (int i = 0; i < ${order + 1}; i++) {
        for (int j = 0; j < ${order + 1}; j++) {
            for (int d = 0; d < ${multipole_dim}; d++) {
                sumreal[i][j][d] = 0.0;
                sumimag[i][j][d] = 0.0;
            }
        }
    }

    for (int src_tri_idx = n_starts[this_obs_n_idx];
            src_tri_idx < n_ends[this_obs_n_idx];
            src_tri_idx++) 
    {
        const int src_tri_rot_clicks = 0;
        ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
        ${prim.tri_info("src", "src_pts", "src_tris", K.needs_srcn, K.surf_curl_src)}
        Real in[9];
        for (int k = 0; k < 9; k++) {
            in[k] = src_in[src_tri_idx * 9 + k];
        }

        for (int iq = 0; iq < ${quad_wts.shape[0]}; iq++) {
            Real srcxhat = quad_pts[iq * 2 + 0];
            Real srcyhat = quad_pts[iq * 2 + 1];
            Real quadw = quad_wts[iq];

            ${prim.basis("src")}
            ${prim.pts_from_basis(
                "y", "src",
                lambda b, d: "src_tri[" + str(b) + "][" + str(d) + "]", 3
            )}

            Real Dx = yx - xx;
            Real Dy = yy - xy; 
            Real Dz = yz - xz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            Real factor = src_jacobian * quadw;
            Real invals[3];
            for (int d = 0; d < 3; d++) {
                invals[d] = 0.0;
                for (int b_src = 0; b_src < 3; b_src++) {
                    invals[d] += factor * in[b_src * 3 + d] * srcb[b_src];
                }
            }


            % if K.surf_curl_src:
                Real src_surf_curl[3][3];
                for (int d = 0; d < 3; d++) {
                    for (int Ij = 0; Ij < 3; Ij++) {
                        src_surf_curl[d][Ij] = 0.0;
                        for (int b_src = 0; b_src < 3; b_src++) {
                            src_surf_curl[d][Ij] += 
                                factor * 
                                bsrc_surf_curl[b_src][Ij]
                                * in[b_src * 3 + d];
                        }
                    }
                }
            % endif

            Real Rsr = 1.0;
            Real Rsi = 0.0;
            for (int mi = 0; mi < ${order} + 1; mi++) {
                ${multipole_sum("mi", "mi", "Rsr", "Rsi")}

                Real Rm2r = 0.0;
                Real Rm2i = 0.0;
                Real Rm1r = Rsr;
                Real Rm1i = Rsi;
                for (int ni = mi; ni < ${order}; ni++) {
                    Real factor = 1.0 / ((ni + 1) * (ni + 1) - mi * mi);
                    Real t1f = (2 * ni + 1) * Dz;
                    Real Rvr = factor * (t1f * Rm1r - r2 * Rm2r);
                    Real Rvi = factor * (t1f * Rm1i - r2 * Rm2i);
                    ${multipole_sum("ni + 1", "mi", "Rvr", "Rvi")}

                    Rm2r = Rm1r;
                    Rm2i = Rm1i;
                    Rm1r = Rvr;
                    Rm1i = Rvi;
                }
                Real Rsrold = Rsr;
                Real Rsiold = Rsi;
                Rsr = (Dx * Rsrold - Dy * Rsiold) / (2 * (mi + 1));
                Rsi = (Dx * Rsiold + Dy * Rsrold) / (2 * (mi + 1));
            }
        }
    }
    ${finish_multipole_sum()}
}

<%def name="get_child_multipoles(d)">
    Real child_real = multipoles[start_idx + ${d} * 2 + 0];
    Real child_imag = multipoles[start_idx + ${d} * 2 + 1];
    if (mi_diff < 0 && mi_diff % 2 == 0) {
        child_imag *= -1;
    } else if (mi_diff < 0 && mi_diff % 2 != 0) {
        child_real *= -1;
    }
    Real realval = child_real * parent_real - child_imag * parent_imag;
    Real imagval = child_real * parent_imag + child_imag * parent_real;
</%def>

<%def name="m2m_sum(nip, mip, real, imag)">
for (int ni = (${nip}); ni <= ${order}; ni++) {
    for (int mi = 0; mi <= ni; mi++) {
        int ni_diff = ni - (${nip});
        for (int mip_sign = -1; mip_sign <= 1; mip_sign += 2) {
            if (${mip} == 0 && mip_sign == -1) {
                continue;
            }
            Real parent_real = ${real};
            Real parent_imag = ${imag};
            int full_mip = (${mip}) * mip_sign;
            if (full_mip < 0 && full_mip % 2 == 0) {
                parent_imag *= -1;
            } else if (full_mip < 0 && full_mip % 2 != 0) {
                parent_real *= -1;
            }

            int mi_diff = mi - full_mip;
            int pos_mi_diff = abs(mi_diff);
            int start_idx = this_src_n_idx * ${multipole_dim * 2 * (order + 1) * (order + 1)}
                + (ni_diff) * ${multipole_dim * 2 * (order + 1)}
                + (pos_mi_diff) * ${multipole_dim * 2};
            {
                ${m2m_core()}
            }
        }
    }
}
</%def>

KERNEL void m2m(
    GLOBAL_MEM Real* multipoles,
    int n_blocks,
    GLOBAL_MEM int* obs_n_idxs,
    GLOBAL_MEM int* obs_src_starts,
    GLOBAL_MEM int* src_n_idxs,
    GLOBAL_MEM Real* n_centers)
{
    const int global_idx = get_global_id(0); 
    const int block_idx = global_idx;
    if (block_idx >= n_blocks) {
        return;
    }
    const int this_obs_n_idx = obs_n_idxs[block_idx];
    const int this_obs_src_start = obs_src_starts[block_idx];
    const int this_obs_src_end = obs_src_starts[block_idx + 1];

    Real xx = n_centers[this_obs_n_idx * 3 + 0];
    Real xy = n_centers[this_obs_n_idx * 3 + 1];
    Real xz = n_centers[this_obs_n_idx * 3 + 2];

    //TODO: Could Kahan summation be helpful?
    Real sumreal[${order + 1}][${order + 1}][${multipole_dim}];
    Real sumimag[${order + 1}][${order + 1}][${multipole_dim}];

    for (int i = 0; i < ${order + 1}; i++) {
        for (int j = 0; j < ${order + 1}; j++) {
            for (int d = 0; d < ${multipole_dim}; d++) {
                sumreal[i][j][d] = 0.0;
                sumimag[i][j][d] = 0.0;
            }
        }
    }

    for (int src_block_idx = this_obs_src_start;
         src_block_idx < this_obs_src_end;
         src_block_idx++) 
    {
        const int this_src_n_idx = src_n_idxs[src_block_idx];

        Real yx = n_centers[this_src_n_idx * 3 + 0];
        Real yy = n_centers[this_src_n_idx * 3 + 1];
        Real yz = n_centers[this_src_n_idx * 3 + 2];

        Real Dx = yx - xx;
        Real Dy = yy - xy; 
        Real Dz = yz - xz;
        Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

        Real Rsr = 1.0;
        Real Rsi = 0.0;
        for (int mip = 0; mip < ${order} + 1; mip++) {
            ${m2m_sum("mip", "mip", "Rsr", "Rsi")}


            Real Rm2r = 0.0;
            Real Rm2i = 0.0;
            Real Rm1r = Rsr;
            Real Rm1i = Rsi;
            for (int nip = mip; nip < ${order}; nip++) {
                Real factor = 1.0 / ((nip + 1) * (nip + 1) - mip * mip);
                Real t1f = (2 * nip + 1) * Dz;
                Real Rvr = factor * (t1f * Rm1r - r2 * Rm2r);
                Real Rvi = factor * (t1f * Rm1i - r2 * Rm2i);
                ${m2m_sum("nip + 1", "mip", "Rvr", "Rvi")}

                Rm2r = Rm1r;
                Rm2i = Rm1i;
                Rm1r = Rvr;
                Rm1i = Rvi;
            }
            Real Rsrold = Rsr;
            Real Rsiold = Rsi;
            Rsr = (Dx * Rsrold - Dy * Rsiold) / (2 * (mip + 1));
            Rsi = (Dx * Rsiold + Dy * Rsrold) / (2 * (mip + 1));
        }
    }
    ${finish_multipole_sum()}
}


KERNEL void m2p_U(
    GLOBAL_MEM Real* out,
    GLOBAL_MEM Real* multipoles,
    GLOBAL_MEM Real* params,
    int n_blocks,
    GLOBAL_MEM int* obs_n_idxs,
    GLOBAL_MEM int* obs_src_starts,
    GLOBAL_MEM int* src_n_idxs,
    GLOBAL_MEM int* obs_n_starts,
    GLOBAL_MEM int* obs_n_ends,
    GLOBAL_MEM Real* pts,
    GLOBAL_MEM int* tris,
    GLOBAL_MEM Real* src_n_centers)
{
    const int global_idx = get_group_id(0); 
    const int worker_idx = get_local_id(0);
    const int block_idx = global_idx;
    const int this_obs_n_idx = obs_n_idxs[block_idx];
    const int this_obs_src_start = obs_src_starts[block_idx];
    const int this_obs_src_end = obs_src_starts[block_idx + 1];

    ${K.constants_code}

    <%
        multipoles_per_cell = (order + 1) ** 2 * multipole_dim * 2
    %>
    LOCAL_MEM Real sh_multipoles[${multipoles_per_cell}];

    int n_start = obs_n_starts[this_obs_n_idx];
    int n_end = obs_n_ends[this_obs_n_idx];
    int n_tris = n_end - n_start;
    int n_outer_idxs = n_tris * ${quad_wts.shape[0]};
    % if ocl_backend:
    int outer_idx_loop_max = n_outer_idxs;
    % else:
    int outer_idx_loop_max = ceil(((float)n_outer_idxs) / ((float)${n_workers_per_block}));
    % endif
    for (int group_outer_idx = 0;
            group_outer_idx < outer_idx_loop_max;
            group_outer_idx++) 
    {
        int outer_idx = group_outer_idx * ${n_workers_per_block} + worker_idx;
        int iq = outer_idx % ${quad_wts.shape[0]};
        int obs_tri_idx = n_start + (outer_idx - iq) / ${quad_wts.shape[0]};

        Real sum[3];
        Real basissum[3][3];
        for (int d1 = 0; d1 < 3; d1++) {
            sum[d1] = 0.0;
            for (int d2 = 0; d2 < 3; d2++) {
                basissum[d1][d2] = 0.0;
            }
        }


        Real obsxhat = quad_pts[iq * 2 + 0];
        Real obsyhat = quad_pts[iq * 2 + 1];
        Real quadw = quad_wts[iq];

        const int obs_tri_rot_clicks = 0;
        ${prim.decl_tri_info("obs", K.needs_obsn, K.surf_curl_obs)}
        if (outer_idx < n_outer_idxs) {
            ${prim.tri_info("obs", "pts", "tris", K.needs_obsn, K.surf_curl_obs)}
        }
        ${prim.basis("obs")}
        ${prim.pts_from_basis(
            "x", "obs",
            lambda b, d: "obs_tri[" + str(b) + "][" + str(d) + "]", 3
        )}

        for (int d1 = 0; d1 < 3; d1++) {
            obsb[d1] *= quadw * obs_jacobian;
            % if K.surf_curl_obs:
                for (int d2 = 0; d2 < 3; d2++) {
                    bobs_surf_curl[d1][d2] *= quadw * obs_jacobian;
                }
            % endif
        }

        for (int src_block_idx = this_obs_src_start;
             src_block_idx < this_obs_src_end;
             src_block_idx++) 
        {
            const int this_src_n_idx = src_n_idxs[src_block_idx];
            LOCAL_BARRIER;
            for (int multipole_idx = worker_idx;
                multipole_idx < ${multipoles_per_cell};
                multipole_idx += ${n_workers_per_block}) 
            {
                int full_arr_idx = this_src_n_idx * ${multipoles_per_cell} + multipole_idx;
                sh_multipoles[multipole_idx] = multipoles[full_arr_idx];
            }
            LOCAL_BARRIER;

            if (outer_idx >= n_outer_idxs) {
                continue;
            }

            //TODO: is it worth preloading centers? probably not, not memory constrained
            Real yx = src_n_centers[this_src_n_idx * 3 + 0];
            Real yy = src_n_centers[this_src_n_idx * 3 + 1];
            Real yz = src_n_centers[this_src_n_idx * 3 + 2];

            Real Dx = xx - yx;
            Real Dy = xy - yy; 
            Real Dz = xz - yz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;
            Real invr2 = 1.0 / r2;

            Real Ssr = sqrt(invr2);
            Real Ssi = 0.0;
            for (int mi = 0; mi < ${order + 1}; mi++) {
                ${m2p_core("mi", "Ssr", "Ssi")}

                Real Sm2r = 0.0;
                Real Sm2i = 0.0;
                Real Sm1r = Ssr;
                Real Sm1i = Ssi;
                for (int ni = mi; ni < ${order}; ni++) {
                    Real t1f = (2 * ni + 1) * Dz;
                    Real t2f = ni * ni - mi * mi;
                    Real Svr = invr2 * (t1f * Sm1r - t2f * Sm2r);
                    Real Svi = invr2 * (t1f * Sm1i - t2f * Sm2i);
                    ${m2p_core("ni + 1", "Svr", "Svi")}

                    Sm2r = Sm1r;
                    Sm2i = Sm1i;
                    Sm1r = Svr;
                    Sm1i = Svi;
                }
                Real Ssrold = Ssr;
                Real Ssiold = Ssi;
                Real F = (2 * mi + 1) * invr2;
                Ssr = F * (Dx * Ssrold - Dy * Ssiold);
                Ssi = F * (Dx * Ssiold + Dy * Ssrold);
            }
        }

        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                basissum[d1][d2] += obsb[d1] * sum[d2];
            }
        }

        if (outer_idx < n_outer_idxs) {
            for (int d1 = 0; d1 < 3; d1++) {
                for (int d2 = 0; d2 < 3; d2++) {
                    % if ocl_backend:
                        out[obs_tri_idx * 9 + d1 * 3 + d2] += basissum[d1][d2];
                    % else:
                        atomicAdd(
                            &out[obs_tri_idx * 9 + d1 * 3 + d2],
                            basissum[d1][d2]
                        );
                    % endif
                }
            }
        }
    }
}
