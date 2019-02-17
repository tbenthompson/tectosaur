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
        Real x${dn(d)} = obs_pts[obs_pt_idx * ${K.spatial_dim} + ${d}];
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
        Real avar4 = avar2 * avar2;
        Real avar6 = avar2 * avar2 * avar2;
        Real avar8 = avar2 * avar2 * avar2 * avar2;
        Real rootavar2 = sqrt(avar2);
        
        Real I1 = -1/(rootavar2*avar2*rhomax);
        Real I2 = -1/(rootavar2*avar2);

        //TODO: Taking advantage of symmetry would make this faster.
        Real I3[3][3] = {{-rootavar2*(avarx * avarx)/(avar6*rhomax), -rootavar2*avarx*avary/(avar6*rhomax), -rootavar2*avarx*avarz/(avar6*rhomax)}, {-rootavar2*avarx*avary/(avar6*rhomax), -rootavar2*(avary * avary)/(avar6*rhomax), -rootavar2*avary*avarz/(avar6*rhomax)}, {-rootavar2*avarx*avarz/(avar6*rhomax), -rootavar2*avary*avarz/(avar6*rhomax), -rootavar2*(avarz * avarz)/(avar6*rhomax)}};

        Real I4[3][3] = {{1/3*(4*avar2*avarx*nsrcx + (avar2*(nsrcx * nsrcx) - 4*(avarx * avarx))*rootavar2)/avar6, 1/3*(2*avar2*avary*nsrcx + 2*avar2*avarx*nsrcy + (avar2*nsrcx*nsrcy - 4*avarx*avary)*rootavar2)/avar6, 1/3*(2*avar2*avarz*nsrcx + 2*avar2*avarx*nsrcz + (avar2*nsrcx*nsrcz - 4*avarx*avarz)*rootavar2)/avar6}, {1/3*(2*avar2*avary*nsrcx + 2*avar2*avarx*nsrcy + (avar2*nsrcx*nsrcy - 4*avarx*avary)*rootavar2)/avar6, 1/3*(4*avar2*avary*nsrcy + (avar2*(nsrcy * nsrcy) - 4*(avary * avary))*rootavar2)/avar6, 1/3*(2*avar2*avarz*nsrcy + 2*avar2*avary*nsrcz + (avar2*nsrcy*nsrcz - 4*avary*avarz)*rootavar2)/avar6}, {1/3*(2*avar2*avarz*nsrcx + 2*avar2*avarx*nsrcz + (avar2*nsrcx*nsrcz - 4*avarx*avarz)*rootavar2)/avar6, 1/3*(2*avar2*avarz*nsrcy + 2*avar2*avary*nsrcz + (avar2*nsrcy*nsrcz - 4*avary*avarz)*rootavar2)/avar6, 1/3*(4*avar2*avarz*nsrcz + (avar2*(nsrcz * nsrcz) - 4*(avarz * avarz))*rootavar2)/avar6}};

        //TODO: There is a huge amount of symmetry here. Only 15 actual different terms. 
        Real I5[3][3][3][3] = {{{{-rootavar2*(avarx*avarx*avarx*avarx)/(avar8*rhomax), -(avarx*avarx*avarx)*avary/(rootavar2*avar6*rhomax), -(avarx*avarx*avarx)*avarz/(rootavar2*avar6*rhomax)}, {-(avarx*avarx*avarx)*avary/(rootavar2*avar6*rhomax), -(avarx*avarx)*(avary*avary)/(rootavar2*avar6*rhomax), -rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax)}, {-(avarx*avarx*avarx)*avarz/(rootavar2*avar6*rhomax), -rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -(avarx*avarx)*(avarz*avarz)/(rootavar2*avar6*rhomax)}}, {{-(avarx*avarx*avarx)*avary/(rootavar2*avar6*rhomax), -(avarx*avarx)*(avary*avary)/(rootavar2*avar6*rhomax), -rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax)}, {-(avarx*avarx)*(avary*avary)/(rootavar2*avar6*rhomax), -avarx*(avary*avary*avary)/(rootavar2*avar6*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax)}, {-rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax)}}, {{-(avarx*avarx*avarx)*avarz/(rootavar2*avar6*rhomax), -rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -(avarx*avarx)*(avarz*avarz)/(rootavar2*avar6*rhomax)}, {-rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax)}, {-(avarx*avarx)*(avarz*avarz)/(rootavar2*avar6*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax), -avarx*(avarz*avarz*avarz)/(rootavar2*avar6*rhomax)}}}, {{{-(avarx*avarx*avarx)*avary/(rootavar2*avar6*rhomax), -(avarx*avarx)*(avary*avary)/(rootavar2*avar6*rhomax), -rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax)}, {-(avarx*avarx)*(avary*avary)/(rootavar2*avar6*rhomax), -avarx*(avary*avary*avary)/(rootavar2*avar6*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax)}, {-rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax)}}, {{-(avarx*avarx)*(avary*avary)/(rootavar2*avar6*rhomax), -avarx*(avary*avary*avary)/(rootavar2*avar6*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax)}, {-avarx*(avary*avary*avary)/(rootavar2*avar6*rhomax), -rootavar2*(avary*avary*avary*avary)/(avar8*rhomax), -(avary*avary*avary)*avarz/(rootavar2*avar6*rhomax)}, {-rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -(avary*avary*avary)*avarz/(rootavar2*avar6*rhomax), -(avary*avary)*(avarz*avarz)/(rootavar2*avar6*rhomax)}}, {{-rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax)}, {-rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -(avary*avary*avary)*avarz/(rootavar2*avar6*rhomax), -(avary*avary)*(avarz*avarz)/(rootavar2*avar6*rhomax)}, {-rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax), -(avary*avary)*(avarz*avarz)/(rootavar2*avar6*rhomax), -avary*(avarz*avarz*avarz)/(rootavar2*avar6*rhomax)}}}, {{{-(avarx*avarx*avarx)*avarz/(rootavar2*avar6*rhomax), -rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -(avarx*avarx)*(avarz*avarz)/(rootavar2*avar6*rhomax)}, {-rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax)}, {-(avarx*avarx)*(avarz*avarz)/(rootavar2*avar6*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax), -avarx*(avarz*avarz*avarz)/(rootavar2*avar6*rhomax)}}, {{-rootavar2*(avarx*avarx)*avary*avarz/(avar8*rhomax), -rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax)}, {-rootavar2*avarx*(avary*avary)*avarz/(avar8*rhomax), -(avary*avary*avary)*avarz/(rootavar2*avar6*rhomax), -(avary*avary)*(avarz*avarz)/(rootavar2*avar6*rhomax)}, {-rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax), -(avary*avary)*(avarz*avarz)/(rootavar2*avar6*rhomax), -avary*(avarz*avarz*avarz)/(rootavar2*avar6*rhomax)}}, {{-(avarx*avarx)*(avarz*avarz)/(rootavar2*avar6*rhomax), -rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax), -avarx*(avarz*avarz*avarz)/(rootavar2*avar6*rhomax)}, {-rootavar2*avarx*avary*(avarz*avarz)/(avar8*rhomax), -(avary*avary)*(avarz*avarz)/(rootavar2*avar6*rhomax), -avary*(avarz*avarz*avarz)/(rootavar2*avar6*rhomax)}, {-avarx*(avarz*avarz*avarz)/(rootavar2*avar6*rhomax), -avary*(avarz*avarz*avarz)/(rootavar2*avar6*rhomax), -rootavar2*(avarz*avarz*avarz*avarz)/(avar8*rhomax)}}}};

        Real I6[3][3][3][3] = {{{{1/15*((2*avar4*(nsrcx*nsrcx*nsrcx*nsrcx) + 18*avar2*(avarx*avarx)*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx*avarx))*rootavar2*rhomax + 8*(avar4*avarx*(nsrcx*nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx)*nsrcx)*rhomax)/(avar8*rhomax), 1/15*((9*avar2*avarx*avary*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx)*avary + (2*avar4*(nsrcx*nsrcx*nsrcx) + 9*avar2*(avarx*avarx)*nsrcx)*nsrcy)*rootavar2*rhomax + 2*(avar4*avary*(nsrcx*nsrcx*nsrcx) + 12*avar2*(avarx*avarx)*avary*nsrcx + (3*avar4*avarx*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx))*nsrcy)*rhomax)/(avar8*rhomax), 1/15*((9*avar2*avarx*avarz*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx)*avarz + (2*avar4*(nsrcx*nsrcx*nsrcx) + 9*avar2*(avarx*avarx)*nsrcx)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*(nsrcx*nsrcx*nsrcx) + 12*avar2*(avarx*avarx)*avarz*nsrcx + (3*avar4*avarx*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx))*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((9*avar2*avarx*avary*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx)*avary + (2*avar4*(nsrcx*nsrcx*nsrcx) + 9*avar2*(avarx*avarx)*nsrcx)*nsrcy)*rootavar2*rhomax + 2*(avar4*avary*(nsrcx*nsrcx*nsrcx) + 12*avar2*(avarx*avarx)*avary*nsrcx + (3*avar4*avarx*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx))*nsrcy)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avarx*nsrcx*(nsrcy*nsrcy) + 4*avarx*(avary*avary)*nsrcx + (avar2*avary*(nsrcx*nsrcx) + 4*(avarx*avarx)*avary)*nsrcy)*rootavar2*rhomax + (3*avar2*(avary*avary)*(nsrcx*nsrcx) + 12*avar2*avarx*avary*nsrcx*nsrcy - 23*(avarx*avarx)*(avary*avary) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcy*nsrcy))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((9*avar2*avarx*avarz*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx)*avarz + (2*avar4*(nsrcx*nsrcx*nsrcx) + 9*avar2*(avarx*avarx)*nsrcx)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*(nsrcx*nsrcx*nsrcx) + 12*avar2*(avarx*avarx)*avarz*nsrcx + (3*avar4*avarx*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avarx*nsrcx*(nsrcz*nsrcz) + 4*avarx*(avarz*avarz)*nsrcx + (avar2*avarz*(nsrcx*nsrcx) + 4*(avarx*avarx)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcx*nsrcx) + 12*avar2*avarx*avarz*nsrcx*nsrcz - 23*(avarx*avarx)*(avarz*avarz) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax)}}, {{1/15*((9*avar2*avarx*avary*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx)*avary + (2*avar4*(nsrcx*nsrcx*nsrcx) + 9*avar2*(avarx*avarx)*nsrcx)*nsrcy)*rootavar2*rhomax + 2*(avar4*avary*(nsrcx*nsrcx*nsrcx) + 12*avar2*(avarx*avarx)*avary*nsrcx + (3*avar4*avarx*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx))*nsrcy)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avarx*nsrcx*(nsrcy*nsrcy) + 4*avarx*(avary*avary)*nsrcx + (avar2*avary*(nsrcx*nsrcx) + 4*(avarx*avarx)*avary)*nsrcy)*rootavar2*rhomax + (3*avar2*(avary*avary)*(nsrcx*nsrcx) + 12*avar2*avarx*avary*nsrcx*nsrcy - 23*(avarx*avarx)*(avary*avary) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcy*nsrcy))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*(4*(avar2*avarx*nsrcx*(nsrcy*nsrcy) + 4*avarx*(avary*avary)*nsrcx + (avar2*avary*(nsrcx*nsrcx) + 4*(avarx*avarx)*avary)*nsrcy)*rootavar2*rhomax + (3*avar2*(avary*avary)*(nsrcx*nsrcx) + 12*avar2*avarx*avary*nsrcx*nsrcy - 23*(avarx*avarx)*(avary*avary) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcy*nsrcy))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((2*avar4*nsrcx*(nsrcy*nsrcy*nsrcy) + 9*avar2*(avary*avary)*nsrcx*nsrcy + 9*avar2*avarx*avary*(nsrcy*nsrcy) - 23*avarx*(avary*avary*avary))*rootavar2*rhomax + 2*(3*avar4*avary*nsrcx*(nsrcy*nsrcy) + avar4*avarx*(nsrcy*nsrcy*nsrcy) + 4*avar2*(avary*avary*avary)*nsrcx + 12*avar2*avarx*(avary*avary)*nsrcy)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}}, {{1/15*((9*avar2*avarx*avarz*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx)*avarz + (2*avar4*(nsrcx*nsrcx*nsrcx) + 9*avar2*(avarx*avarx)*nsrcx)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*(nsrcx*nsrcx*nsrcx) + 12*avar2*(avarx*avarx)*avarz*nsrcx + (3*avar4*avarx*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avarx*nsrcx*(nsrcz*nsrcz) + 4*avarx*(avarz*avarz)*nsrcx + (avar2*avarz*(nsrcx*nsrcx) + 4*(avarx*avarx)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcx*nsrcx) + 12*avar2*avarx*avarz*nsrcx*nsrcz - 23*(avarx*avarx)*(avarz*avarz) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax)}, {1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*(4*(avar2*avarx*nsrcx*(nsrcz*nsrcz) + 4*avarx*(avarz*avarz)*nsrcx + (avar2*avarz*(nsrcx*nsrcx) + 4*(avarx*avarx)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcx*nsrcx) + 12*avar2*avarx*avarz*nsrcx*nsrcz - 23*(avarx*avarx)*(avarz*avarz) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((2*avar4*nsrcx*(nsrcz*nsrcz*nsrcz) + 9*avar2*(avarz*avarz)*nsrcx*nsrcz + 9*avar2*avarx*avarz*(nsrcz*nsrcz) - 23*avarx*(avarz*avarz*avarz))*rootavar2*rhomax + 2*(3*avar4*avarz*nsrcx*(nsrcz*nsrcz) + avar4*avarx*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcx + 12*avar2*avarx*(avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}}}, {{{1/15*((9*avar2*avarx*avary*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx)*avary + (2*avar4*(nsrcx*nsrcx*nsrcx) + 9*avar2*(avarx*avarx)*nsrcx)*nsrcy)*rootavar2*rhomax + 2*(avar4*avary*(nsrcx*nsrcx*nsrcx) + 12*avar2*(avarx*avarx)*avary*nsrcx + (3*avar4*avarx*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx))*nsrcy)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avarx*nsrcx*(nsrcy*nsrcy) + 4*avarx*(avary*avary)*nsrcx + (avar2*avary*(nsrcx*nsrcx) + 4*(avarx*avarx)*avary)*nsrcy)*rootavar2*rhomax + (3*avar2*(avary*avary)*(nsrcx*nsrcx) + 12*avar2*avarx*avary*nsrcx*nsrcy - 23*(avarx*avarx)*(avary*avary) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcy*nsrcy))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*(4*(avar2*avarx*nsrcx*(nsrcy*nsrcy) + 4*avarx*(avary*avary)*nsrcx + (avar2*avary*(nsrcx*nsrcx) + 4*(avarx*avarx)*avary)*nsrcy)*rootavar2*rhomax + (3*avar2*(avary*avary)*(nsrcx*nsrcx) + 12*avar2*avarx*avary*nsrcx*nsrcy - 23*(avarx*avarx)*(avary*avary) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcy*nsrcy))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((2*avar4*nsrcx*(nsrcy*nsrcy*nsrcy) + 9*avar2*(avary*avary)*nsrcx*nsrcy + 9*avar2*avarx*avary*(nsrcy*nsrcy) - 23*avarx*(avary*avary*avary))*rootavar2*rhomax + 2*(3*avar4*avary*nsrcx*(nsrcy*nsrcy) + avar4*avarx*(nsrcy*nsrcy*nsrcy) + 4*avar2*(avary*avary*avary)*nsrcx + 12*avar2*avarx*(avary*avary)*nsrcy)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}}, {{1/15*(4*(avar2*avarx*nsrcx*(nsrcy*nsrcy) + 4*avarx*(avary*avary)*nsrcx + (avar2*avary*(nsrcx*nsrcx) + 4*(avarx*avarx)*avary)*nsrcy)*rootavar2*rhomax + (3*avar2*(avary*avary)*(nsrcx*nsrcx) + 12*avar2*avarx*avary*nsrcx*nsrcy - 23*(avarx*avarx)*(avary*avary) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcy*nsrcy))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((2*avar4*nsrcx*(nsrcy*nsrcy*nsrcy) + 9*avar2*(avary*avary)*nsrcx*nsrcy + 9*avar2*avarx*avary*(nsrcy*nsrcy) - 23*avarx*(avary*avary*avary))*rootavar2*rhomax + 2*(3*avar4*avary*nsrcx*(nsrcy*nsrcy) + avar4*avarx*(nsrcy*nsrcy*nsrcy) + 4*avar2*(avary*avary*avary)*nsrcx + 12*avar2*avarx*(avary*avary)*nsrcy)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((2*avar4*nsrcx*(nsrcy*nsrcy*nsrcy) + 9*avar2*(avary*avary)*nsrcx*nsrcy + 9*avar2*avarx*avary*(nsrcy*nsrcy) - 23*avarx*(avary*avary*avary))*rootavar2*rhomax + 2*(3*avar4*avary*nsrcx*(nsrcy*nsrcy) + avar4*avarx*(nsrcy*nsrcy*nsrcy) + 4*avar2*(avary*avary*avary)*nsrcx + 12*avar2*avarx*(avary*avary)*nsrcy)*rhomax)/(avar8*rhomax), 1/15*((2*avar4*(nsrcy*nsrcy*nsrcy*nsrcy) + 18*avar2*(avary*avary)*(nsrcy*nsrcy) - 23*(avary*avary*avary*avary))*rootavar2*rhomax + 8*(avar4*avary*(nsrcy*nsrcy*nsrcy) + 4*avar2*(avary*avary*avary)*nsrcy)*rhomax)/(avar8*rhomax), 1/15*((9*avar2*avary*avarz*(nsrcy*nsrcy) - 23*(avary*avary*avary)*avarz + (2*avar4*(nsrcy*nsrcy*nsrcy) + 9*avar2*(avary*avary)*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*(nsrcy*nsrcy*nsrcy) + 12*avar2*(avary*avary)*avarz*nsrcy + (3*avar4*avary*(nsrcy*nsrcy) + 4*avar2*(avary*avary*avary))*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((9*avar2*avary*avarz*(nsrcy*nsrcy) - 23*(avary*avary*avary)*avarz + (2*avar4*(nsrcy*nsrcy*nsrcy) + 9*avar2*(avary*avary)*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*(nsrcy*nsrcy*nsrcy) + 12*avar2*(avary*avary)*avarz*nsrcy + (3*avar4*avary*(nsrcy*nsrcy) + 4*avar2*(avary*avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avary*nsrcy*(nsrcz*nsrcz) + 4*avary*(avarz*avarz)*nsrcy + (avar2*avarz*(nsrcy*nsrcy) + 4*(avary*avary)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcy*nsrcy) + 12*avar2*avary*avarz*nsrcy*nsrcz - 23*(avary*avary)*(avarz*avarz) + (2*avar4*(nsrcy*nsrcy) + 3*avar2*(avary*avary))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax)}}, {{1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((9*avar2*avary*avarz*(nsrcy*nsrcy) - 23*(avary*avary*avary)*avarz + (2*avar4*(nsrcy*nsrcy*nsrcy) + 9*avar2*(avary*avary)*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*(nsrcy*nsrcy*nsrcy) + 12*avar2*(avary*avary)*avarz*nsrcy + (3*avar4*avary*(nsrcy*nsrcy) + 4*avar2*(avary*avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avary*nsrcy*(nsrcz*nsrcz) + 4*avary*(avarz*avarz)*nsrcy + (avar2*avarz*(nsrcy*nsrcy) + 4*(avary*avary)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcy*nsrcy) + 12*avar2*avary*avarz*nsrcy*nsrcz - 23*(avary*avary)*(avarz*avarz) + (2*avar4*(nsrcy*nsrcy) + 3*avar2*(avary*avary))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax)}, {1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avary*nsrcy*(nsrcz*nsrcz) + 4*avary*(avarz*avarz)*nsrcy + (avar2*avarz*(nsrcy*nsrcy) + 4*(avary*avary)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcy*nsrcy) + 12*avar2*avary*avarz*nsrcy*nsrcz - 23*(avary*avary)*(avarz*avarz) + (2*avar4*(nsrcy*nsrcy) + 3*avar2*(avary*avary))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((2*avar4*nsrcy*(nsrcz*nsrcz*nsrcz) + 9*avar2*(avarz*avarz)*nsrcy*nsrcz + 9*avar2*avary*avarz*(nsrcz*nsrcz) - 23*avary*(avarz*avarz*avarz))*rootavar2*rhomax + 2*(3*avar4*avarz*nsrcy*(nsrcz*nsrcz) + avar4*avary*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcy + 12*avar2*avary*(avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}}}, {{{1/15*((9*avar2*avarx*avarz*(nsrcx*nsrcx) - 23*(avarx*avarx*avarx)*avarz + (2*avar4*(nsrcx*nsrcx*nsrcx) + 9*avar2*(avarx*avarx)*nsrcx)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*(nsrcx*nsrcx*nsrcx) + 12*avar2*(avarx*avarx)*avarz*nsrcx + (3*avar4*avarx*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx*avarx))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avarx*nsrcx*(nsrcz*nsrcz) + 4*avarx*(avarz*avarz)*nsrcx + (avar2*avarz*(nsrcx*nsrcx) + 4*(avarx*avarx)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcx*nsrcx) + 12*avar2*avarx*avarz*nsrcx*nsrcz - 23*(avarx*avarx)*(avarz*avarz) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax)}, {1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*(4*(avar2*avarx*nsrcx*(nsrcz*nsrcz) + 4*avarx*(avarz*avarz)*nsrcx + (avar2*avarz*(nsrcx*nsrcx) + 4*(avarx*avarx)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcx*nsrcx) + 12*avar2*avarx*avarz*nsrcx*nsrcz - 23*(avarx*avarx)*(avarz*avarz) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((2*avar4*nsrcx*(nsrcz*nsrcz*nsrcz) + 9*avar2*(avarz*avarz)*nsrcx*nsrcz + 9*avar2*avarx*avarz*(nsrcz*nsrcz) - 23*avarx*(avarz*avarz*avarz))*rootavar2*rhomax + 2*(3*avar4*avarz*nsrcx*(nsrcz*nsrcz) + avar4*avarx*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcx + 12*avar2*avarx*(avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}}, {{1/15*((3*avar2*avary*avarz*(nsrcx*nsrcx) + 6*avar2*avarx*avarz*nsrcx*nsrcy - 23*(avarx*avarx)*avary*avarz + (6*avar2*avarx*avary*nsrcx + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(8*avar2*avarx*avary*avarz*nsrcx + (avar4*avarz*(nsrcx*nsrcx) + 4*avar2*(avarx*avarx)*avarz)*nsrcy + (avar4*avary*(nsrcx*nsrcx) + 2*avar4*avarx*nsrcx*nsrcy + 4*avar2*(avarx*avarx)*avary)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((6*avar2*avary*avarz*nsrcx*nsrcy + 3*avar2*avarx*avarz*(nsrcy*nsrcy) - 23*avarx*(avary*avary)*avarz + (2*avar4*nsrcx*(nsrcy*nsrcy) + 3*avar2*(avary*avary)*nsrcx + 6*avar2*avarx*avary*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*nsrcx*(nsrcy*nsrcy) + 4*avar2*(avary*avary)*avarz*nsrcx + 8*avar2*avarx*avary*avarz*nsrcy + (2*avar4*avary*nsrcx*nsrcy + avar4*avarx*(nsrcy*nsrcy) + 4*avar2*avarx*(avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((9*avar2*avary*avarz*(nsrcy*nsrcy) - 23*(avary*avary*avary)*avarz + (2*avar4*(nsrcy*nsrcy*nsrcy) + 9*avar2*(avary*avary)*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(avar4*avarz*(nsrcy*nsrcy*nsrcy) + 12*avar2*(avary*avary)*avarz*nsrcy + (3*avar4*avary*(nsrcy*nsrcy) + 4*avar2*(avary*avary*avary))*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avary*nsrcy*(nsrcz*nsrcz) + 4*avary*(avarz*avarz)*nsrcy + (avar2*avarz*(nsrcy*nsrcy) + 4*(avary*avary)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcy*nsrcy) + 12*avar2*avary*avarz*nsrcy*nsrcz - 23*(avary*avary)*(avarz*avarz) + (2*avar4*(nsrcy*nsrcy) + 3*avar2*(avary*avary))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax)}, {1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avary*nsrcy*(nsrcz*nsrcz) + 4*avary*(avarz*avarz)*nsrcy + (avar2*avarz*(nsrcy*nsrcy) + 4*(avary*avary)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcy*nsrcy) + 12*avar2*avary*avarz*nsrcy*nsrcz - 23*(avary*avary)*(avarz*avarz) + (2*avar4*(nsrcy*nsrcy) + 3*avar2*(avary*avary))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((2*avar4*nsrcy*(nsrcz*nsrcz*nsrcz) + 9*avar2*(avarz*avarz)*nsrcy*nsrcz + 9*avar2*avary*avarz*(nsrcz*nsrcz) - 23*avary*(avarz*avarz*avarz))*rootavar2*rhomax + 2*(3*avar4*avarz*nsrcy*(nsrcz*nsrcz) + avar4*avary*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcy + 12*avar2*avary*(avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}}, {{1/15*(4*(avar2*avarx*nsrcx*(nsrcz*nsrcz) + 4*avarx*(avarz*avarz)*nsrcx + (avar2*avarz*(nsrcx*nsrcx) + 4*(avarx*avarx)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcx*nsrcx) + 12*avar2*avarx*avarz*nsrcx*nsrcz - 23*(avarx*avarx)*(avarz*avarz) + (2*avar4*(nsrcx*nsrcx) + 3*avar2*(avarx*avarx))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((2*avar4*nsrcx*(nsrcz*nsrcz*nsrcz) + 9*avar2*(avarz*avarz)*nsrcx*nsrcz + 9*avar2*avarx*avarz*(nsrcz*nsrcz) - 23*avarx*(avarz*avarz*avarz))*rootavar2*rhomax + 2*(3*avar4*avarz*nsrcx*(nsrcz*nsrcz) + avar4*avarx*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcx + 12*avar2*avarx*(avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((3*avar2*(avarz*avarz)*nsrcx*nsrcy - 23*avarx*avary*(avarz*avarz) + (2*avar4*nsrcx*nsrcy + 3*avar2*avarx*avary)*(nsrcz*nsrcz) + 6*(avar2*avary*avarz*nsrcx + avar2*avarx*avarz*nsrcy)*nsrcz)*rootavar2*rhomax + 2*(4*avar2*avary*(avarz*avarz)*nsrcx + 4*avar2*avarx*(avarz*avarz)*nsrcy + (avar4*avary*nsrcx + avar4*avarx*nsrcy)*(nsrcz*nsrcz) + 2*(avar4*avarz*nsrcx*nsrcy + 4*avar2*avarx*avary*avarz)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*(4*(avar2*avary*nsrcy*(nsrcz*nsrcz) + 4*avary*(avarz*avarz)*nsrcy + (avar2*avarz*(nsrcy*nsrcy) + 4*(avary*avary)*avarz)*nsrcz)*rootavar2*rhomax + (3*avar2*(avarz*avarz)*(nsrcy*nsrcy) + 12*avar2*avary*avarz*nsrcy*nsrcz - 23*(avary*avary)*(avarz*avarz) + (2*avar4*(nsrcy*nsrcy) + 3*avar2*(avary*avary))*(nsrcz*nsrcz))*rhomax)/(rootavar2*avar6*rhomax), 1/15*((2*avar4*nsrcy*(nsrcz*nsrcz*nsrcz) + 9*avar2*(avarz*avarz)*nsrcy*nsrcz + 9*avar2*avary*avarz*(nsrcz*nsrcz) - 23*avary*(avarz*avarz*avarz))*rootavar2*rhomax + 2*(3*avar4*avarz*nsrcy*(nsrcz*nsrcz) + avar4*avary*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcy + 12*avar2*avary*(avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}, {1/15*((2*avar4*nsrcx*(nsrcz*nsrcz*nsrcz) + 9*avar2*(avarz*avarz)*nsrcx*nsrcz + 9*avar2*avarx*avarz*(nsrcz*nsrcz) - 23*avarx*(avarz*avarz*avarz))*rootavar2*rhomax + 2*(3*avar4*avarz*nsrcx*(nsrcz*nsrcz) + avar4*avarx*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcx + 12*avar2*avarx*(avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((2*avar4*nsrcy*(nsrcz*nsrcz*nsrcz) + 9*avar2*(avarz*avarz)*nsrcy*nsrcz + 9*avar2*avary*avarz*(nsrcz*nsrcz) - 23*avary*(avarz*avarz*avarz))*rootavar2*rhomax + 2*(3*avar4*avarz*nsrcy*(nsrcz*nsrcz) + avar4*avary*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcy + 12*avar2*avary*(avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax), 1/15*((2*avar4*(nsrcz*nsrcz*nsrcz*nsrcz) + 18*avar2*(avarz*avarz)*(nsrcz*nsrcz) - 23*(avarz*avarz*avarz*avarz))*rootavar2*rhomax + 8*(avar4*avarz*(nsrcz*nsrcz*nsrcz) + 4*avar2*(avarz*avarz*avarz)*nsrcz)*rhomax)/(avar8*rhomax)}}}};
        
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

        Real rrrr_dot_nsrc_nobs[3][3][3];
        for (int dobs = 0; dobs < 3; dobs++) {
            for (int dsrc = 0; dsrc < 3; dsrc++) {
                rrrr_dot_nsrc_nobs[0][dobs][dsrc] = 0.0;
                rrrr_dot_nsrc_nobs[1][dobs][dsrc] = 0.0;
                rrrr_dot_nsrc_nobs[2][dobs][dsrc] = 0.0;
                % for j in range(3):
                    % for k in range(3):
                    {
                        Real I5v = I5[dobs][dsrc][${j}][${k}] * nsrc${dn(j)} * nobs${dn(k)};
                        Real I6v = I6[dobs][dsrc][${j}][${k}] * nsrc${dn(j)} * nobs${dn(k)};
                        rrrr_dot_nsrc_nobs[0][dobs][dsrc] += -I6v*cos(t) - I6v*sin(t) + I5v;
                        rrrr_dot_nsrc_nobs[1][dobs][dsrc] += I6v*cos(t);
                        rrrr_dot_nsrc_nobs[2][dobs][dsrc] += I6v*sin(t);
                    }
                    % endfor
                % endfor
            }
        }

        for (int b = 0; b < 3; b++) {
            % for dobs in range(3):
                % for dsrc in range(3):
                {
                    Real Kval = CsH0*CsH2*nobs${dn(dobs)}*T0basis[b]*nsrc${dn(dsrc)};

                    Kval += CsH0*CsH1*nsrc${dn(dobs)}*T0basis[b]*nobs${dn(dsrc)};

                    Kval += 3*CsH0*CsH1*nobs${dn(dobs)}*rr_dot_nsrc[b][${dsrc}];

                    Kval += CsH0*CsH3*nsrc${dn(dobs)}*rr_dot_nobs[b][${dsrc}];

                    Kval += 3*CsH0*CsH1*nsrc${dn(dsrc)}*rr_dot_nobs[b][${dobs}];

                    Kval += CsH0*CsH3*T1basis[b][${dobs}][${dsrc}]*mn;

                    Kval += 3*CsH0*nu*nobs${dn(dsrc)}*rr_dot_nsrc[b][${dobs}];

                    Kval += -15*CsH0*CsH1*rrrr_dot_nsrc_nobs[b][${dobs}][${dsrc}];

                    % if dobs == dsrc:
                        % for j in range(3):
                            Kval += 3*nu*CsH0*nobs${dn(j)}*rr_dot_nsrc[b][${j}];
                        % endfor
                        Kval += CsH0*CsH1*mn*T0basis[b];
                    % endif

                    result_temp[${dobs} * 9 + b * 3 + ${dsrc}] += quadw * Kval;
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
