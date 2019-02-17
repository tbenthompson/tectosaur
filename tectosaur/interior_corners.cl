<%
from tectosaur.kernels import kernels
K = kernels['elasticH3']
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>
${cluda_preamble}

#define Real ${float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

${prim.geometry_fncs()}

<%def name="setup_pair()">
    const int i = get_global_id(0);
    const int pair_idx = i + start_idx;
    if (pair_idx >= end_idx) {
        return;
    }
</%def>

KERNEL
void interior_corners(GLOBAL_MEM Real* result, 
    int n_quad_pts, GLOBAL_MEM Real* quad_pts, GLOBAL_MEM Real* quad_wts,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* obs_ns,
    GLOBAL_MEM Real* src_pts, GLOBAL_MEM int* src_tris,
    GLOBAL_MEM int* pairs_list, int start_idx, int end_idx, 
    GLOBAL_MEM Real* params)
{
    ${setup_pair()}

    const int obs_pt_idx = pairs_list[pair_idx * 3];
    const int src_tri_idx = pairs_list[pair_idx * 3 + 1];
    const int src_tri_rot_clicks = pairs_list[pair_idx * 3 + 2];

    ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
    ${prim.tri_info("src", "src_pts", "src_tris", K.needs_srcn, K.surf_curl_src)}

    Real result_temp[27];
    Real kahanC[27];

    for (int iresult = 0; iresult < 27; iresult++) {
        result_temp[iresult] = 0;
        kahanC[iresult] = 0;
    }

    % for d in range(K.spatial_dim):
        Real nobs${dn(d)} = obs_ns[obs_pt_idx * ${K.spatial_dim} + ${d}];
    % endfor


    ${K.constants_code}

    Real v1[3];
    Real v2[3];
    % for d in range(3):
        v1[${d}] = src_tri[1][${d}] - src_tri[0][${d}];
        v2[${d}] = src_tri[2][${d}] - src_tri[0][${d}];
    % endfor

    for (int iq = 0; iq < n_quad_pts; iq++) {
        Real t = quad_pts[iq];
        Real quadw = quad_wts[iq];
        Real rhomax = 1.0 / (cos(t) + sin(t));

        % for d in range(3):
            Real avar${dn(d)} = v1[${d}] * cos(t) + v2[${d}] * sin(t);
        % endfor
        Real avar2 = avarx * avarx + avary * avary + avarz * avarz;
        Real avar = sqrt(avar2);
        Real avar3 = avar2 * avar;
        Real avar4 = avar3 * avar;
        Real avar5 = avar4 * avar;
        Real avar6 = avar5 * avar;
        Real avar7 = avar6 * avar;
        Real avar8 = avar7 * avar;
        
        //TODO: Taking advantage of symmetry in I3, I4, I5, I6  would make this faster.
        Real I1 = -1/(avar3*rhomax);
        Real I2 = (log((Real)2.0) + log(avar) + log(rhomax) - 1)/avar3;
        Real I3[3][3] = {{-(avarx*avarx)/(avar5*rhomax), -avarx*avary/(avar5*rhomax), -avarx*avarz/(avar5*rhomax)}, {-avarx*avary/(avar5*rhomax), -(avary*avary)/(avar5*rhomax), -avary*avarz/(avar5*rhomax)}, {-avarx*avarz/(avar5*rhomax), -avary*avarz/(avar5*rhomax), -(avarz*avarz)/(avar5*rhomax)}};
        Real I4[3][3] = {{1/3*(avar2*(nsrcx*nsrcx) - 4*avar*avarx*nsrcx + (avarx*avarx)*(3*log((Real)2.0) + 3*log(avar) - 4) + 3*(avarx*avarx)*log(rhomax))/avar5, -1/3*(2*avar*avary*nsrcx - avarx*avary*(3*log((Real)2.0) + 3*log(avar) - 4) - 3*avarx*avary*log(rhomax) - (avar2*nsrcx - 2*avar*avarx)*nsrcy)/avar5, -1/3*(2*avar*avarz*nsrcx - avarx*avarz*(3*log((Real)2.0) + 3*log(avar) - 4) - 3*avarx*avarz*log(rhomax) - (avar2*nsrcx - 2*avar*avarx)*nsrcz)/avar5}, {-1/3*(2*avar*avary*nsrcx - avarx*avary*(3*log((Real)2.0) + 3*log(avar) - 4) - 3*avarx*avary*log(rhomax) - (avar2*nsrcx - 2*avar*avarx)*nsrcy)/avar5, 1/3*(avar2*(nsrcy*nsrcy) - 4*avar*avary*nsrcy + (avary*avary)*(3*log((Real)2.0) + 3*log(avar) - 4) + 3*(avary*avary)*log(rhomax))/avar5, -1/3*(2*avar*avarz*nsrcy - avary*avarz*(3*log((Real)2.0) + 3*log(avar) - 4) - 3*avary*avarz*log(rhomax) - (avar2*nsrcy - 2*avar*avary)*nsrcz)/avar5}, {-1/3*(2*avar*avarz*nsrcx - avarx*avarz*(3*log((Real)2.0) + 3*log(avar) - 4) - 3*avarx*avarz*log(rhomax) - (avar2*nsrcx - 2*avar*avarx)*nsrcz)/avar5, -1/3*(2*avar*avarz*nsrcy - avary*avarz*(3*log((Real)2.0) + 3*log(avar) - 4) - 3*avary*avarz*log(rhomax) - (avar2*nsrcy - 2*avar*avary)*nsrcz)/avar5, 1/3*(avar2*(nsrcz*nsrcz) - 4*avar*avarz*nsrcz + (avarz*avarz)*(3*log((Real)2.0) + 3*log(avar) - 4) + 3*(avarz*avarz)*log(rhomax))/avar5}};
        Real T0basis[3] = {
            -I2*cos(t) - I2*sin(t) + I1,
            I2*cos(t),
            I2*sin(t)
        };

        Real T1basis[3][3][3];
        for (int dobs = 0; dobs < 3; dobs++) {
            for (int dsrc = 0; dsrc < 3; dsrc++) {
                Real I3v = I3[dobs][dsrc];
                Real I4v = I4[dobs][dsrc];
                T1basis[0][dobs][dsrc] = -I4v*cos(t) - I4v*sin(t) + I3v;
                T1basis[1][dobs][dsrc] = I4v*cos(t);
                T1basis[2][dobs][dsrc] = I4v*sin(t);
            }
        }

        Real mn = nobsx * nsrcx + nobsy * nsrcy + nobsz * nsrcz;

        Real rr_dot_nsrc[3][3];
        Real rr_dot_nobs[3][3];
        for (int b = 0; b < 3; b++) {
            for (int d = 0; d < 3; d++) {
                rr_dot_nsrc[b][d] = 0;
                rr_dot_nobs[b][d] = 0;
                % for j in range(3):
                    rr_dot_nsrc[b][d] += nsrc${dn(j)}*T1basis[b][${j}][d];
                    rr_dot_nobs[b][d] += nobs${dn(j)}*T1basis[b][${j}][d];
                % endfor
            }
        }

        for (int b = 0; b < 3; b++) {
            % for dobs in range(3):
                % for dsrc in range(3):
                {
                    Real Kval = 0.0;
                    //Real Kval = CsH0*CsH2*nobs${dn(dobs)}*T0basis[b]*nsrc${dn(dsrc)};

                    //Kval += CsH0*CsH1*nsrc${dn(dobs)}*T0basis[b]*nobs${dn(dsrc)};

                    //Kval += 3*CsH0*CsH1*nobs${dn(dobs)}*rr_dot_nsrc[b][${dsrc}];

                    //Kval += CsH0*CsH3*nsrc${dn(dobs)}*rr_dot_nobs[b][${dsrc}];

                    //Kval += 3*CsH0*CsH1*nsrc${dn(dsrc)}*rr_dot_nobs[b][${dobs}];

                    Kval += CsH0*CsH3*T1basis[b][${dobs}][${dsrc}]*mn;

                    //Kval += 3*CsH0*nu*nobs${dn(dsrc)}*rr_dot_nsrc[b][${dobs}];

                    //Kval += -15*CsH0*CsH1*rrrr_dot_nsrc_nobs[b][${dobs}][${dsrc}];

                    % if dobs == dsrc:
                        % for j in range(3):
                            //Kval += 3*nu*CsH0*nobs${dn(j)}*rr_dot_nsrc[b][${j}];
                        % endfor
                        Kval += CsH0*CsH1*mn*T0basis[b];
                    % endif

                    int idx = ${dobs} * 9 + b * 3 + ${dsrc};
                    Real val = quadw * Kval;
                    Real y = val - kahanC[idx];
                    Real t = result_temp[idx] + y;
                    kahanC[idx] = (t - result_temp[idx]) - y;
                    result_temp[idx] = t;
                }
                % endfor
            % endfor
        }
         
        //nobs[dobs]*MT[dsrc] = 
        //nobs[dobs]*(Q*CsH2*nsrc[dsrc] + A*CsH1*Dor[dsrc])
        //MT: CsH0*CsH2*nobs[dobs]*invr3*nsrc[dsrc] +  3*CsH0*CsH1*nobs[dobs]*invr3*(nsrcx*Dorx+nsrcy*Dory+nsrcz*Dorz)*Dor[dsrc])
        //
        
        //NT: nsrc[dobs]*CsH0*invr3*CsH1*nobs[dsrc] + CsH0*CsH3*nsrc[dobs]*Dor[dsrc]*(nobsx * Dorx + nobsy * Dory + nobsz * Dorz)

        //DT term 1: 3*CsH0*CsH1*nsrc[dsrc]*Dor[dobs]*(nobsx * Dorx + nobsy * Dory + nobsz * Dorz)
        //
        //DT term 2: CsH0*invr3*CsH3*Dor[dobs]*Dor[dsrc]*mn
        //DT term 3: 3*CsH0*nu*invr3*nobs[dsrc]*Dor[dobs]*rn
        //DT term 4: -15*CsH0*invr3*Dor[dsrc]*Dor[dobs]*rm*rn
        //
        //ST term 1: 3*CsH0*nu*invr3*rn*rm
        //ST term 2: CsH0*invr3*CsH1*mn

        //Real Dorx = invr * Dx;
        //Real Dory = invr * Dy;
        //Real Dorz = invr * Dz;

        //Real rn = nsrcx * Dorx + nsrcy * Dory + nsrcz * Dorz;
        //Real rm = nobsx * Dorx + nobsy * Dory + nobsz * Dorz;
        //Real mn = nobsx * nsrcx + nobsy * nsrcy + nobsz * nsrcz;

        //Real Q = CsH0 * invr3;
        //Real A = CsH0 * invr3 * 3 * rn;
        //Real B = CsH0 * invr3 * CsH1;
        //Real C = CsH0 * invr3 * CsH3;

        //Real MTx = Q*CsH2*nsrcx + A*CsH1*Dorx;
        //Real MTy = Q*CsH2*nsrcy + A*CsH1*Dory;
        //Real MTz = Q*CsH2*nsrcz + A*CsH1*Dorz;

        //Real NTx = B*nobsx + C*Dorx*rm;
        //Real NTy = B*nobsy + C*Dory*rm;
        //Real NTz = B*nobsz + C*Dorz*rm;

        //Real DTx = B*3*nsrcx*rm + C*Dorx*mn + A*(nu*nobsx - 5*Dorx*rm);
        //Real DTy = B*3*nsrcy*rm + C*Dory*mn + A*(nu*nobsy - 5*Dory*rm);
        //Real DTz = B*3*nsrcz*rm + C*Dorz*mn + A*(nu*nobsz - 5*Dorz*rm);

        //Real ST = A*nu*rm + B*mn;

        //Karr[0] = nsrcx*NTx + nobsx*MTx + Dorx*DTx + ST;
        //Karr[1] = nsrcx*NTy + nobsx*MTy + Dorx*DTy;
        //Karr[2] = nsrcx*NTz + nobsx*MTz + Dorx*DTz;
        //Karr[3] = nsrcy*NTx + nobsy*MTx + Dory*DTx;
        //Karr[4] = nsrcy*NTy + nobsy*MTy + Dory*DTy + ST;
        //Karr[5] = nsrcy*NTz + nobsy*MTz + Dory*DTz;
        //Karr[6] = nsrcz*NTx + nobsz*MTx + Dorz*DTx;
        //Karr[7] = nsrcz*NTy + nobsz*MTy + Dorz*DTy;
        //Karr[8] = nsrcz*NTz + nobsz*MTz + Dorz*DTz + ST;

    }

    int src_derot[3];
    for (size_t i = 0; i < 3; i++) {
        src_derot[i] = positive_mod(-src_tri_rot_clicks + i, 3);
    }
    for (int d1 = 0; d1 < 3; d1++) {
        for (int b2 = 0; b2 < 3; b2++) {
            for (int d2 = 0; d2 < 3; d2++) {
                int out_idx = d1 * 9 + b2 * 3 + d2;
                int in_idx = d1 * 9 + src_derot[b2] * 3 + d2;
                Real val = src_jacobian * result_temp[in_idx];
                result[i * 27 + out_idx] = val;
            }
        }
    }
}

//v1 = tri[1] - tri[0]
//v2 = tri[2] - tri[0]
//a = v1 * cos(t) + v2 * sin(t)
//A2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
//I3 = 1/3*(2*A2*a[j]*n[i] + 2*A2*a[i]*n[j] + (A2*n[i]*n[j] - 4*a[i]*a[j])*sqrt(A2))/(A2 ** 3)
//I4 = -sqrt(A2)*a[i]*a[j]/((A2 ** 3)*rhomaxvar)
//
//Real invr = rsqrt(r2);
//Real invr2 = invr * invr;
//Real invr3 = invr2 * invr;
//Real Dorx = invr * Dx;
//Real Dory = invr * Dy;
//Real Dorz = invr * Dz;
//
//Real rn = nsrcx * Dorx + nsrcy * Dory + nsrcz * Dorz;
//Real rm = nobsx * Dorx + nobsy * Dory + nobsz * Dorz;
//Real mn = nobsx * nsrcx + nobsy * nsrcy + nobsz * nsrcz;
//
//Real Q = CsH0 * invr3;
//Real A = Q * 3 * rn;
//Real B = Q * CsH1;
//Real C = Q * CsH3;
//
//Real I3;
//Real I4;
//
//Real MTx = Q*CsH2*nsrcx + A*CsH1*Dorx;
//Real MTy = Q*CsH2*nsrcy + A*CsH1*Dory;
//Real MTz = Q*CsH2*nsrcz + A*CsH1*Dorz;
//
//Real NTx = B*nobsx + C*Dorx*rm;
//Real NTy = B*nobsy + C*Dory*rm;
//Real NTz = B*nobsz + C*Dorz*rm;
//
//Real DTx = B*3*nsrcx*rm + C*Dorx*mn + A*(nu*nobsx - 5*Dorx*rm);
//Real DTy = B*3*nsrcy*rm + C*Dory*mn + A*(nu*nobsy - 5*Dory*rm);
//Real DTz = B*3*nsrcz*rm + C*Dorz*mn + A*(nu*nobsz - 5*Dorz*rm);
//
//Real ST = A*nu*rm + B*mn;
//
//Karr[0] = nsrcx*NTx + nobsx*MTx + Dorx*DTx + ST;
//Karr[1] = nsrcx*NTy + nobsx*MTy + Dorx*DTy;
//Karr[2] = nsrcx*NTz + nobsx*MTz + Dorx*DTz;
//Karr[3] = nsrcy*NTx + nobsy*MTx + Dory*DTx;
//Karr[4] = nsrcy*NTy + nobsy*MTy + Dory*DTy + ST;
//Karr[5] = nsrcy*NTz + nobsy*MTz + Dory*DTz;
//Karr[6] = nsrcz*NTx + nobsz*MTx + Dorz*DTx;
//Karr[7] = nsrcz*NTy + nobsz*MTy + Dorz*DTy;
//Karr[8] = nsrcz*NTz + nobsz*MTz + Dorz*DTz + ST;
