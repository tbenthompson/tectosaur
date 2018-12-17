<%!
def dn(dim):
    return ['x', 'y', 'z'][dim]
from tectosaur.kernels import kernels
%>
${cluda_preamble}

#define Real ${gpu_float_type}

<%namespace name="prim" file="../integral_primitives.cl"/>

${prim.geometry_fncs()}

CONSTANT Real quad_pts[${quad_pts.size}] = {${str(quad_pts.flatten().tolist())[1:-1]}};
CONSTANT Real quad_wts[${quad_wts.size}] = {${str(quad_wts.flatten().tolist())[1:-1]}};

<%def name="multipole_sum(n, m, real, imag)">
    % for d in range(3):
        sumreal[${n}][${m}][${d}] += ${real} * invals[${d}];
        sumimag[${n}][${m}][${d}] += ${imag} * invals[${d}];
    % endfor
    sumreal[${n}][${m}][3] += ${real} * in_dot_D;
    sumimag[${n}][${m}][3] += ${imag} * in_dot_D;
</%def>

<%def name="finish_multipole_sum()">
    for (int i = 0; i < ${order + 1}; i++) {
        for (int j = 0; j < ${order + 1}; j++) {
            for (int d = 0; d < 4; d++) {
                int idx = this_obs_n_idx * ${4 * 2 * (order + 1) * (order + 1)}
                    + i * ${4 * 2 * (order + 1)}
                    + j * 4 * 2
                    + d * 2;
                multipoles[idx] = sumreal[i][j][d];
                multipoles[idx + 1] = sumimag[i][j][d];
            }
        }
    }
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

    ${kernels['elasticU3'].constants_code}

    const int obs_tri_rot_clicks = 0;
    ${prim.decl_tri_info("obs", False, False)}
    ${prim.tri_info("obs", "obs_pts", "obs_tris", False, False)}

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
            ${prim.decl_tri_info("src", False, False)}
            ${prim.tri_info("src", "src_pts", "src_tris", False, False)}

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

                    ${prim.call_vector_code(kernels['elasticU3'])}

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

KERNEL void p2m(
    GLOBAL_MEM Real* multipoles,
    GLOBAL_MEM Real* src_in,
    GLOBAL_MEM int* n_idxs,
    GLOBAL_MEM Real* n_centers,
    GLOBAL_MEM int* n_starts,
    GLOBAL_MEM int* n_ends,
    GLOBAL_MEM Real* pts,
    GLOBAL_MEM int* tris)
{
    const int global_idx = get_global_id(0); 
    const int block_idx = global_idx;
    const int this_obs_n_idx = n_idxs[block_idx];

    Real xx = n_centers[this_obs_n_idx * 3 + 0];
    Real xy = n_centers[this_obs_n_idx * 3 + 1];
    Real xz = n_centers[this_obs_n_idx * 3 + 2];

    Real sumreal[${order + 1}][${order + 1}][4];
    Real sumimag[${order + 1}][${order + 1}][4];

    for (int i = 0; i < ${order + 1}; i++) {
        for (int j = 0; j < ${order + 1}; j++) {
            for (int d = 0; d < 4; d++) {
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
        ${prim.decl_tri_info("src", False, False)}
        ${prim.tri_info("src", "pts", "tris", False, False)}
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
                    invals[d] += factor * src_in[src_tri_idx * 9 + b_src * 3 + d] * srcb[b_src];
                }
            }

            Real in_dot_D = invals[0] * Dx + invals[1] * Dy + invals[2] * Dz;

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
            int start_idx = this_src_n_idx * ${4 * 2 * (order + 1) * (order + 1)}
                + (ni_diff) * ${4 * 2 * (order + 1)}
                + (pos_mi_diff) * 4 * 2;
            %for d in range(3):
            {
                ${get_child_multipoles(d)}
                sumreal[ni][mi][${d}] += realval;
                sumimag[ni][mi][${d}] += imagval;

                sumreal[ni][mi][3] += D${dn(d)} * realval;
                sumimag[ni][mi][3] += D${dn(d)} * imagval;
            }
            % endfor

            ${get_child_multipoles(3)}
            sumreal[ni][mi][3] += realval;
            sumimag[ni][mi][3] += imagval;
        }
    }
}
</%def>

KERNEL void m2m(
    GLOBAL_MEM Real* multipoles,
    GLOBAL_MEM int* obs_n_idxs,
    GLOBAL_MEM int* obs_src_starts,
    GLOBAL_MEM int* src_n_idxs,
    GLOBAL_MEM Real* n_centers)
{
    const int global_idx = get_global_id(0); 
    const int block_idx = global_idx;
    const int this_obs_n_idx = obs_n_idxs[block_idx];
    const int this_obs_src_start = obs_src_starts[block_idx];
    const int this_obs_src_end = obs_src_starts[block_idx + 1];

    Real xx = n_centers[this_obs_n_idx * 3 + 0];
    Real xy = n_centers[this_obs_n_idx * 3 + 1];
    Real xz = n_centers[this_obs_n_idx * 3 + 2];

    //TODO: Kahan summation could be helpful?
    Real sumreal[${order + 1}][${order + 1}][4];
    Real sumimag[${order + 1}][${order + 1}][4];

    for (int i = 0; i < ${order + 1}; i++) {
        for (int j = 0; j < ${order + 1}; j++) {
            for (int d = 0; d < 4; d++) {
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

<%def name="out_sum_S_U(n, real, imag)">
    ${out_sum_dS("(" + n + ")", "mi", "(" + real + ")", "(" + imag + ")")}
    for (int d = 0; d < 3; d++) {
        int idx = this_src_n_idx * ${4 * 2 * (order + 1) * (order + 1)}
            + (${n}) * ${4 * 2 * (order + 1)}
            + mi * 4 * 2
            + d * 2;
        Real Rr = multipoles[idx];
        Real Ri = multipoles[idx + 1];
        for (int bi = 0; bi < 3; bi++) {
            sum[bi][d] += CsU0 * obsb[bi] * (Rr * ${real} + Ri * ${imag});
        }
    }
    if (mi > 0) {
        Real mult = 1.0;
        if (mi % 2 == 0) {
            mult = -1.0;
        }
        ${out_sum_dS("(" + n + ")", "(-mi)", "(-mult * " + real + ")", "(mult * " + imag + ")")}

        for (int d = 0; d < 3; d++) {
            int idx = this_src_n_idx * ${4 * 2 * (order + 1) * (order + 1)}
                + (${n}) * ${4 * 2 * (order + 1)}
                + mi * 4 * 2
                + d * 2;
            Real Rr = -mult * multipoles[idx];
            Real Ri = mult * multipoles[idx + 1];
            for (int bi = 0; bi < 3; bi++) {
                sum[bi][d] += CsU0 * obsb[bi] * (Rr * -mult * ${real} + Ri * mult * ${imag});
            }
        }
    }
</%def>

<%def name="out_sum_dS(n, m, real, imag)">
    int dSn = (${n}) - 1;
    int dSmm1 = (${m}) - 1;
    int dSmp1 = (${m}) + 1;

    if (abs(dSmm1) <= dSn) {
        Real dSr1 = -0.5 * (${real});    
        Real dSi1 = -0.5 * (${imag});        
        ${dS_sum("dSn", "dSmm1", 0, "dSr1", "dSi1")}
        Real dSr2 = -0.5 * (${imag});    
        Real dSi2 = 0.5 * (${real});        
        ${dS_sum("dSn", "dSmm1", 1, "dSr2", "dSi2")}
    }

    if (abs(dSmp1) <= dSn) {
        Real dSr1 = 0.5 * (${real});        
        Real dSi1 = 0.5 * (${imag});        
        ${dS_sum("dSn", "dSmp1", 0, "dSr1", "dSi1")}
        Real dSr2 = -0.5 * (${imag});        
        Real dSi2 =  0.5 * (${real});        
        ${dS_sum("dSn", "dSmp1", 1, "dSr2", "dSi2")}
    }

    if (abs(${m}) <= dSn) {
        Real dSr3 = -(${real});
        Real dSi3 = -(${imag});
        ${dS_sum("dSn", m, 2, "dSr3", "dSi3")}
    }
</%def>

<%def name="dS_sum(dsn, dsm, d1, real, imag)">
{
    Real multRR = 1.0;
    Real multRI = 1.0;
    if (${dsm} < 0) {
        if (${dsm} % 2 == 0) {
            multRI = -1.0;
        } else {
            multRR = -1.0;
        }
    }
    int base_idx = this_src_n_idx * ${4 * 2 * (order + 1) * (order + 1)}
        + (${dsn}) * ${4 * 2 * (order + 1)}
        + abs(${dsm}) * 4 * 2;
    % for d2 in range(3):
    {
        int idx = base_idx + ${d2} * 2;
        Real Rr = multRR * multipoles[idx];
        Real Ri = multRI * multipoles[idx + 1];
        Real v = D${dn(d2)} * (Rr * ${real} + Ri * ${imag});
        for (int bi = 0; bi < 3; bi++) {
            sum[bi][${d1}] -= CsU1 * obsb[bi] * v;
        }
    }
    % endfor
    int idx = base_idx + 3 * 2;
    Real Rr = multRR * multipoles[idx];
    Real Ri = multRI * multipoles[idx + 1];
    Real v =  Rr * ${real} + Ri * ${imag};
    for (int bi = 0; bi < 3; bi++) {
        sum[bi][${d1}] += CsU1 * obsb[bi] * v;
    }
}
</%def>


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
    const int global_idx = get_global_id(0); 
    if (global_idx >= n_blocks) {
        return;
    }
    const int worker_idx = get_local_id(1);
    const int block_idx = global_idx;
    const int this_obs_n_idx = obs_n_idxs[block_idx];
    const int this_obs_src_start = obs_src_starts[block_idx];
    const int this_obs_src_end = obs_src_starts[block_idx + 1];

    const Real G = params[0];
    const Real nu = params[1];
    const Real CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
    const Real CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));


    int n_start = obs_n_starts[this_obs_n_idx];
    int n_end = obs_n_ends[this_obs_n_idx];
    int n_tris = n_end - n_start;
    for (int outer_idx = worker_idx;
            outer_idx < n_tris * ${quad_wts.shape[0]};
            outer_idx += ${n_workers_per_block}) 
    {
        int iq = outer_idx % ${quad_wts.shape[0]};
        int obs_tri_idx = n_start + (outer_idx - iq) / ${quad_wts.shape[0]};

        const int obs_tri_rot_clicks = 0;
        ${prim.decl_tri_info("obs", False, False)}
        ${prim.tri_info("obs", "pts", "tris", False, False)}

        Real sum[3][3];
        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                sum[d1][d2] = 0.0;
            }
        }

        Real obsxhat = quad_pts[iq * 2 + 0];
        Real obsyhat = quad_pts[iq * 2 + 1];
        Real quadw = quad_wts[iq];

        ${prim.basis("obs")}
        ${prim.pts_from_basis(
            "x", "obs",
            lambda b, d: "obs_tri[" + str(b) + "][" + str(d) + "]", 3
        )}

        for (int d = 0; d < 3; d++) {
            obsb[d] *= quadw * obs_jacobian;
        }

        for (int src_block_idx = this_obs_src_start;
             src_block_idx < this_obs_src_end;
             src_block_idx++) 
        {
            const int this_src_n_idx = src_n_idxs[src_block_idx];

            //TODO: is it worth preloading centers?
            Real yx = src_n_centers[this_src_n_idx * 3 + 0];
            Real yy = src_n_centers[this_src_n_idx * 3 + 1];
            Real yz = src_n_centers[this_src_n_idx * 3 + 2];
            // TODO: load multipoles here

            Real Dx = xx - yx;
            Real Dy = xy - yy; 
            Real Dz = xz - yz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            Real Ssr = 1.0 / sqrt(r2);
            Real Ssi = 0.0;
            for (int mi = 0; mi < ${order + 1}; mi++) {
                ${out_sum_S_U("mi", "Ssr", "Ssi")}

                Real Sm2r = 0.0;
                Real Sm2i = 0.0;
                Real Sm1r = Ssr;
                Real Sm1i = Ssi;
                for (int ni = mi; ni < ${order}; ni++) {
                    Real F = 1.0 / r2;
                    Real t1f = (2 * ni + 1) * Dz;
                    Real t2f = ni * ni - mi * mi;
                    Real Svr = F * (t1f * Sm1r - t2f * Sm2r);
                    Real Svi = F * (t1f * Sm1i - t2f * Sm2i);
                    ${out_sum_S_U("ni + 1", "Svr", "Svi")}

                    Sm2r = Sm1r;
                    Sm2i = Sm1i;
                    Sm1r = Svr;
                    Sm1i = Svi;
                }
                Real Ssrold = Ssr;
                Real Ssiold = Ssi;
                Real F = (2 * mi + 1) / r2;
                Ssr = F * (Dx * Ssrold - Dy * Ssiold);
                Ssi = F * (Dx * Ssiold + Dy * Ssrold);
            }
        }

        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                atomicAdd(&out[obs_tri_idx * 9 + d1 * 3 + d2], sum[d1][d2]);
            }
        }
    }
}
