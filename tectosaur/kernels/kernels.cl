//TODO: refactor to put all kernels here!
<%!
import numpy as np
e = [[[int((i - j) * (j - k) * (k - i) / 2) for k in range(3)]
    for j in range(3)] for i in range(3)]
kronecker = np.array([[1,0,0],[0,1,0],[0,0,1]])
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

<%def name="elasticRTA3_setup(obs_src)">
    Real invr = rsqrt(r2);
    Real invr3 = invr * invr * invr;
    Real Q1 = CsRT1 * invr;
    Real Q2 = CsRT0 * invr3;
    Real Q3 = CsRT2 * invr3;
    Real Q3nD = Q3 * (n${obs_src}x * Dx + n${obs_src}y * Dy + n${obs_src}z * Dz);
</%def>

<%def name="Gimk(p, m, j)">
    Real A = Q1 * ${e[m][p][j]};
    Real B = 0.0; 
    % for It in range(3):
        B += ${e[m][It][j]} * D${dn(It)};
    % endfor
    Real Gimk = A + Q2 * D${dn(p)} * B;
</%def>

<%def name="elasticRTA3_tensor(L, R)">
    % for d_obs in range(3):
    % for d_src in range(3):
    <%
        L_d = d_obs if L == 'obs' else d_src
        R_d = d_src if L == 'obs' else d_obs
    %>
    for (int b_${R} = 0; b_${R} < 3; b_${R}++) {
    {
        Real Kval = 0.0;
        %for Ij in range(3):
        {
            ${Gimk(L_d, Ij, R_d)}
            Kval += Gimk * b${R}_surf_curl[b_${R}][${Ij}];
        }
        % endfor
        % if d_obs == d_src:
            Kval += Q3nD * ${R}b[b_${R}] * ${1 if L == 'obs' else -1};
        % endif
        for (int b_${L} = 0; b_${L} < 3; b_${L}++) {
            Real val = quadw * ${L}b[b_${L}] * Kval;
            int idx = b_obs * 27 + ${d_obs} * 9 + b_src * 3 + ${d_src};
            result_temp[idx] += val;
        }
    }
    }
    % endfor
    % endfor
</%def>

<%def name="elasticRT3_tensor()">
    ${elasticRTA3_setup('src')}
    ${elasticRTA3_tensor('obs', 'src')}
</%def>

<%def name="elasticRA3_tensor()">
    ${elasticRTA3_setup('obs')}
    ${elasticRTA3_tensor('src', 'obs')}
</%def>

<%def name="elasticRH3_tensor()">
    const Real invr2 = 1.0 / r2;
    const Real invr = sqrt(invr2);
    % for d_obs in range(3):
    % for d_src in range(3):
    % for It in range(3):
    % for Im in range(3):
    {
        //k is d_obs, j is d_src, 
        Real Kval = quadw * CsRH0 * invr * (
            -2 * D${dn(d_obs)} * D${dn(d_src)} * invr2 * ${kronecker[It][Im]}
            + CsRH1 * ${kronecker[d_obs][It] * kronecker[d_src][Im]}
            + CsRH2 * ${kronecker[d_obs][Im] * kronecker[d_src][It]}
            - 2 * ${kronecker[d_obs][d_src] * kronecker[It][Im]}
        );
        for (int b_src = 0; b_src < 3; b_src++) {
        for (int b_obs = 0; b_obs < 3; b_obs++) {
            int idx = b_obs * 27 + ${d_obs} * 9 + b_src * 3 + ${d_src};
            result_temp[idx] += Kval 
                * bobs_surf_curl[b_obs][${It}] 
                * bsrc_surf_curl[b_src][${Im}];;
        }
        }
    }
    % endfor
    % endfor
    % endfor
    % endfor
</%def>

<%def name="elasticRT3_vector()">
//TODO: refactor out
    Real src_surf_curl[3][3];
    for (int Ij = 0; Ij < 3; Ij++) {
        for (int d = 0; d < 3; d++) {
            src_surf_curl[d][Ij] = 0.0;
            for (int b_src = 0; b_src < 3; b_src++) {
                src_surf_curl[d][Ij] += 
                    bsrc_surf_curl[b_src][Ij]
                    * in[b_src * 3 + d];
            }
        }
    }

    ${elasticRTA3_setup('src')}

    % for d_obs in range(3):
    % for d_src in range(3):
    % for Ij in range(3):
    {
        ${Gimk(d_obs, Ij, d_src)}
        sum${dn(d_obs)} += Gimk * src_surf_curl[${d_src}][${Ij}];;
    }
    % endfor
    % endfor
    % endfor

    % for d in range(3):
        sum${dn(d)} += Q3nD * in${dn(d)};
    % endfor
</%def>

<%def name="elasticRA3_vector()">
    ${elasticRTA3_setup('obs')}

    % for d_obs in range(3):
    % for d_src in range(3):
    % for Ij in range(3):
    {
        ${Gimk(d_src, Ij, d_obs)}
        Real Kval = factor * Gimk * in${dn(d_src)};
        for (int b_obs = 0; b_obs < 3; b_obs++) {
            sum[b_obs * 3 + ${d_obs}] += Kval * bobs_surf_curl[b_obs][${Ij}];
        }
    }
    % endfor
    % endfor
    % endfor

    % for d in range(3):
        sum${dn(d)} -= Q3nD * in${dn(d)};
    % endfor
</%def>

<%def name="elasticRH3_vector()">
    const Real invr2 = 1.0 / r2;
    const Real invr = sqrt(invr2);
    % for d_obs in range(3):
    % for d_src in range(3):
    % for It in range(3):
    % for Im in range(3):
    {
        //k is d_obs, j is d_src, 
        Real Kval = factor * CsRH0 * invr * (
            -2 * D${dn(d_obs)} * D${dn(d_src)} * invr2 * ${kronecker[It][Im]}
            + CsRH1 * ${kronecker[d_obs][It] * kronecker[d_src][Im]}
            + CsRH2 * ${kronecker[d_obs][Im] * kronecker[d_src][It]}
            - 2 * ${kronecker[d_obs][d_src] * kronecker[It][Im]}
        );
        for (int b_src = 0; b_src < 3; b_src++) {
        for (int b_obs = 0; b_obs < 3; b_obs++) {
            sum[b_obs * 3 + ${d_obs}] += Kval 
                * bobs_surf_curl[b_obs][${It}] 
                * bsrc_surf_curl[b_src][${Im}] * in[b_src * 3 + ${d_src}];
        }
        }
    }
    % endfor
    % endfor
    % endfor
    % endfor
</%def>

<%def name="elasticRH3_vector2()">
    const Real invr2 = 1.0 / r2;
    const Real invr = sqrt(invr2);
    Real F = factor * CsRH0 * invr;

    Real src_surf_curl[3][3];
    for (int Ij = 0; Ij < 3; Ij++) {
        for (int d = 0; d < 3; d++) {
            src_surf_curl[d][Ij] = 0.0;
            for (int b_src = 0; b_src < 3; b_src++) {
                src_surf_curl[d][Ij] += 
                    bsrc_surf_curl[b_src][Ij]
                    * in[b_src * 3 + d];
            }
        }
    }

    Real B = 0.0;
    % for d_src in range(3):
        B += src_surf_curl[${d_src}][${d_src}];
    % endfor
    B *= CsRH1;

    for (int b_obs = 0; b_obs < 3; b_obs++) {
        Real A = 0.0;
        % for d_src in range(3):
            A += D${dn(d_src)} * (
                src_surf_curl[${d_src}][0] * bobs_surf_curl[b_obs][0]
                + src_surf_curl[${d_src}][1] * bobs_surf_curl[b_obs][1]
                + src_surf_curl[${d_src}][2] * bobs_surf_curl[b_obs][2]
            );
        % endfor

        % for d_obs in range(3):
        {
            Real C = src_surf_curl[0][${d_obs}] * bobs_surf_curl[b_obs][0]
                + src_surf_curl[1][${d_obs}] * bobs_surf_curl[b_obs][1]
                + src_surf_curl[2][${d_obs}] * bobs_surf_curl[b_obs][2];

            Real D = src_surf_curl[${d_obs}][0] * bobs_surf_curl[b_obs][0]
                + src_surf_curl[${d_obs}][1] * bobs_surf_curl[b_obs][1]
                + src_surf_curl[${d_obs}][2] * bobs_surf_curl[b_obs][2];

            sum[b_obs * 3 + ${d_obs}] += F * (
                -2 * D${dn(d_obs)} * invr2 * A
                + bobs_surf_curl[b_obs][${d_obs}] * B 
                + CsRH2 * C 
                - 2 * D
            );
        }
        % endfor
    }
</%def>
