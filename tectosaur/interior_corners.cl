${cluda_preamble}

#define Real ${float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

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
        Real x${dn(d)} = obs_pts[obs_pt_idx * ${K.spatial_dim} + ${d}];
        Real nobs${dn(d)} = obs_ns[obs_pt_idx * ${K.spatial_dim} + ${d}];
    % endfor

    ${K.constants_code}

    Real v1[3];
    Real v2[3];
    % for d in range(3):
        Real v1d = src_tri[1][d] - src_tri[0][d];
        Real v2d = src_tri[2][d] - src_tri[0][d];
    % endfor

    Real
    
    for (int iq = 0; iq < n_quad_pts; iq++) {
        Real thetahat = quad_pts[iq];
        Real quadw = quad_wts[iq];
        Real rhomaxvar = 1.0 / (cos(thetahat) + sin(thetahat));

        % for d in range(3):
            Real avar${dn(d)} = v1[${d}] * cos(thetahat) + v2[${d}] * sin(thetahat);
        % endfor
        Real avar2 = avarx * avarx + avary * avary + avarz * avarz;
        Real rootavar2 = sqrt(avar2)
        
        Real I1 = -1/(rootavar2*avar2*rhomax);
        Real I2 = -1/(rootavar2*avar2);

        Real I3[3][3] = {{-rootavar2*(avarx * avarx)/(avar6*rhomax), -rootavar2*avarx*avary/(avar6*rhomax), -rootavar2*avarx*avarz/(avar6*rhomax)}, {-rootavar2*avarx*avary/(avar6*rhomax), -rootavar2*(avary * avary)/(avar6*rhomax), -rootavar2*avary*avarz/(avar6*rhomax)}, {-rootavar2*avarx*avarz/(avar6*rhomax), -rootavar2*avary*avarz/(avar6*rhomax), -rootavar2*(avarz * avarz)/(avar6*rhomax)}};

        Real I4[3][3] = {{1/3*(4*avar2*avarx*nsrcx + (avar2*(nsrcx * nsrcx) - 4*(avarx * avarx))*rootavar2)/avar6, 1/3*(2*avar2*avary*nsrcx + 2*avar2*avarx*nsrcy + (avar2*nsrcx*nsrcy - 4*avarx*avary)*rootavar2)/avar6, 1/3*(2*avar2*avarz*nsrcx + 2*avar2*avarx*nsrcz + (avar2*nsrcx*nsrcz - 4*avarx*avarz)*rootavar2)/avar6}, {1/3*(2*avar2*avary*nsrcx + 2*avar2*avarx*nsrcy + (avar2*nsrcx*nsrcy - 4*avarx*avary)*rootavar2)/avar6, 1/3*(4*avar2*avary*nsrcy + (avar2*(nsrcy * nsrcy) - 4*(avary * avary))*rootavar2)/avar6, 1/3*(2*avar2*avarz*nsrcy + 2*avar2*avary*nsrcz + (avar2*nsrcy*nsrcz - 4*avary*avarz)*rootavar2)/avar6}, {1/3*(2*avar2*avarz*nsrcx + 2*avar2*avarx*nsrcz + (avar2*nsrcx*nsrcz - 4*avarx*avarz)*rootavar2)/avar6, 1/3*(2*avar2*avarz*nsrcy + 2*avar2*avary*nsrcz + (avar2*nsrcy*nsrcz - 4*avary*avarz)*rootavar2)/avar6, 1/3*(4*avar2*avarz*nsrcz + (avar2*(nsrcz * nsrcz) - 4*(avarz * avarz))*rootavar2)/avar6}};

        % for dobs in range(3):
            % for dsrc in range(3):

                Real Kval = nsrc${dn(dobs)}*NT${dn(dsrc)};
                Kval += nobs${dn(dobs)}*MT${dn(dsrc)};
                Kval += Dor${dn(dobs)}*DT${dn(dsrc)};

                Karr[${dobs * 3 + dsrc}] = Kval
                % if dobs == dsrc:
                    Karr[${dobs * 3 + dsrc}] += ST;
                % endif
            % endfor
        % endfor
         
        //Real Dorx = invr * Dx;
        //Real Dory = invr * Dy;
        //Real Dorz = invr * Dz;

        //Real rn = nsrcx * Dorx + nsrcy * Dory + nsrcz * Dorz;
        //Real rm = nobsx * Dorx + nobsy * Dory + nobsz * Dorz;
        //Real mn = nobsx * nsrcx + nobsy * nsrcy + nobsz * nsrcz;

        //Real Q = CsH0 * invr3;
        //Real A = Q * 3 * rn;
        //Real B = Q * CsH1;
        //Real C = Q * CsH3;

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

        //Real T0basis0 = -I2*cos(t) - I2*sin(t) + I1;
        //Real T0basis1 = I2*cos(t);
        //Real T0basis2 = I2*sin(t);

        //Real T1basis0 = -I4*cos(t) - I4*sin(t) + I3;
        //Real T1basis1 = I4*cos(t);
        //Real T1basis2 = I4*sin(t);
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
