<%!
e = [[[int((i - j) * (j - k) * (k - i) / 2) for k in range(3)]
    for j in range(3)] for i in range(3)]
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

<%def name="elasticRT3_vector()">
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

    Real invr = rsqrt(r2);
    Real invr3 = invr * invr * invr;
    Real Q1 = CsRT1 * invr;
    Real Q2 = CsRT0 * invr3;
    Real Q3 = CsRT2 * invr3;
    Real Q3nD = Q3 * (nsrcx * Dx + nsrcy * Dy + nsrcz * Dz);

    % for d_obs in range(3):
    % for d_src in range(3):
    % for Ij in range(3):
    {
        Real A = Q1 * ${e[Ij][d_obs][d_src]};
        Real B = 0.0; 
        % for Ip in range(3):
            B += ${e[Ij][Ip][d_src]} * D${dn(Ip)};
        % endfor
        B = Q2 * D${dn(d_obs)} * B;
        sum${dn(d_obs)} += (A + B) * src_surf_curl[${d_src}][${Ij}];;
    }
    % endfor
    % endfor
    % endfor

    % for d in range(3):
        sum${dn(d)} += Q3nD * in${dn(d)};
    % endfor
</%def>

<%def name="elasticRT3_tensor()">
    Real invr = rsqrt(r2);
    Real invr3 = invr * invr * invr;
    Real Q1 = CsRT1 * invr;
    Real Q2 = CsRT0 * invr3;
    Real Q3 = CsRT2 * invr3;
    Real Q3nD = Q3 * (nsrcx * Dx + nsrcy * Dy + nsrcz * Dz);

    % for d_obs in range(3):
    for (int b_src = 0; b_src < 3; b_src++) {
    % for d_src in range(3):
    {
        Real Kval = 0.0;
        %for Ij in range(3):
        {
            Real A = Q1 * ${e[Ij][d_obs][d_src]};
            Real B = 0.0;
            % for Ip in range(3):
                B += ${e[Ij][Ip][d_src]} * D${dn(Ip)};
            % endfor
            B = Q2 * D${dn(d_obs)} * B;
            Kval += (A + B) * bsrc_surf_curl[b_src][${Ij}];
        }
        % endfor
        % if d_obs == d_src:
            Kval += Q3nD * srcb[b_src];
        % endif
        for (int b_obs = 0; b_obs < 3; b_obs++) {
            Real val = quadw * obsb[b_obs] * Kval;
            int idx = b_obs * 27 + ${d_obs} * 9 + b_src * 3 + ${d_src};
            result_temp[idx] += val;
        }
    }
    % endfor
    }
    % endfor
</%def>
