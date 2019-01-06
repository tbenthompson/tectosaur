import attr

@attr.s
class Kernel:
    name = attr.ib()
    spatial_dim = attr.ib()
    tensor_dim = attr.ib()
    needs_obsn = attr.ib()
    needs_srcn = attr.ib()
    scale_type = attr.ib()
    sm_power = attr.ib()
    flip_negate = attr.ib()
    constants_code = attr.ib()
    vector_code = attr.ib()
    tensor_code = attr.ib()
    surf_curl_obs = attr.ib(default = False)
    surf_curl_src = attr.ib(default = False)
    multipole_dim = attr.ib(default = 0)

elasticU = Kernel(
    'elasticU3', 3, 3, False, False, -3, 1, False,
    '''
    const Real G = params[0];
    const Real nu = params[1];
    const Real CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
    const Real CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
    ''',
    '''
    Real invr = rsqrt(r2);
    Real Q1 = CsU0 * invr;
    Real Q2 = CsU1 * invr / r2;
    Real ddi = Dx*inx + Dy*iny + Dz*inz;
    sumx += Q1*inx + Q2*Dx*ddi;
    sumy += Q1*iny + Q2*Dy*ddi;
    sumz += Q1*inz + Q2*Dz*ddi;
    ''',
    '''
    Real invr = rsqrt(r2);
    Real Q1 = CsU0 * invr;
    Real Q2 = CsU1 * invr / r2;
    Karr[0] = Q2*Dx*Dx + Q1;
    Karr[1] = Q2*Dx*Dy;
    Karr[2] = Q2*Dx*Dz;
    Karr[3] = Q2*Dy*Dx;
    Karr[4] = Q2*Dy*Dy + Q1;
    Karr[5] = Q2*Dy*Dz;
    Karr[6] = Q2*Dz*Dx;
    Karr[7] = Q2*Dz*Dy;
    Karr[8] = Q2*Dz*Dz + Q1;
    ''',
    multipole_dim = 7
)

TA_const_code = '''
    const Real nu = params[1];
    const Real CsT0 = (1-2.0*nu)/(8.0*M_PI*(1.0-nu));
    const Real CsT1 = 3.0/(8.0*M_PI*(1.0-nu));
    '''

def TA_args(k_name):
    return dict(
        minus_or_plus = '-' if k_name is 'T' else '+',
        plus_or_minus = '+' if k_name is 'T' else '-',
        n_name = 'nsrc' if k_name is 'T' else 'nobs'
    )

def TA_vector_code(k_name):
    return '''
        Real invr = rsqrt(r2);
        Real invr2 = invr * invr;
        Real invr3 = invr2 * invr;

        Real rn = {n_name}x * Dx + {n_name}y * Dy + {n_name}z * Dz;

        Real A = {plus_or_minus}CsT0 * invr3;
        Real C = {minus_or_plus}CsT1 * invr3 * invr2;

        Real rnddi = C * rn * (Dx*inx + Dy*iny + Dz*inz);

        Real nxdy = {n_name}x*Dy-{n_name}y*Dx;
        Real nzdx = {n_name}z*Dx-{n_name}x*Dz;
        Real nzdy = {n_name}z*Dy-{n_name}y*Dz;

        sumx += A*(
            - rn * inx
            {minus_or_plus} nxdy * iny
            {plus_or_minus} nzdx * inz)
            + Dx*rnddi;
        sumy += A*(
            {plus_or_minus} nxdy * inx
            - rn * iny
            {plus_or_minus} nzdy * inz)
            + Dy*rnddi;
        sumz += A*(
            {minus_or_plus} nzdx * inx
            {minus_or_plus} nzdy * iny
            - rn * inz)
            + Dz*rnddi;
    '''.format(**TA_args(k_name))

def TA_tensor_code(k_name):
    return '''
        Real invr = rsqrt(r2);
        Real invr2 = invr * invr;
        Real invr3 = invr2 * invr;

        Real rn = {n_name}x * Dx + {n_name}y * Dy + {n_name}z * Dz;

        Real A = {plus_or_minus}CsT0 * invr3;
        Real C = {minus_or_plus}CsT1 * invr3 * invr2;

        Real nxdy = {n_name}x*Dy-{n_name}y*Dx;
        Real nzdx = {n_name}z*Dx-{n_name}x*Dz;
        Real nzdy = {n_name}z*Dy-{n_name}y*Dz;

        Karr[0] = A * -rn                  + C*Dx*rn*Dx;
        Karr[1] = A * {minus_or_plus}nxdy + C*Dx*rn*Dy;
        Karr[2] = A * {plus_or_minus}nzdx + C*Dx*rn*Dz;
        Karr[3] = A * {plus_or_minus}nxdy + C*Dy*rn*Dx;
        Karr[4] = A * -rn                  + C*Dy*rn*Dy;
        Karr[5] = A * {plus_or_minus}nzdy + C*Dy*rn*Dz;
        Karr[6] = A * {minus_or_plus}nzdx + C*Dz*rn*Dx;
        Karr[7] = A * {minus_or_plus}nzdy + C*Dz*rn*Dy;
        Karr[8] = A * -rn                  + C*Dz*rn*Dz;
    '''.format(**TA_args(k_name))

elasticT = Kernel(
    'elasticT3', 3, 3, False, True, -2, 0, True,
    TA_const_code, TA_vector_code('T'), TA_tensor_code('T'),
)

elasticA = Kernel(
    'elasticA3', 3, 3, True, False, -2, 0, True,
    TA_const_code, TA_vector_code('A'), TA_tensor_code('A'),
)

elasticH = Kernel(
    'elasticH3', 3, 3, True, True, -1, -1, False,
    '''
    const Real G = params[0];
    const Real nu = params[1];
    const Real CsH0 = G/(4*M_PI*(1-nu));
    const Real CsH1 = 1-2*nu;
    const Real CsH2 = -1+4*nu;
    const Real CsH3 = 3*nu;
    ''',
    '''
    Real invr = rsqrt(r2);
    Real invr2 = invr * invr;
    Real invr3 = invr2 * invr;

    Real rn = invr*(nsrcx * Dx + nsrcy * Dy + nsrcz * Dz);
    Real rm = invr*(nobsx * Dx + nobsy * Dy + nobsz * Dz);
    Real mn = nobsx * nsrcx + nobsy * nsrcy + nobsz * nsrcz;

    Real sn = inx*nsrcx + iny*nsrcy + inz*nsrcz;
    Real sd = invr*(inx*Dx + iny*Dy + inz*Dz);
    Real sm = inx*nobsx + iny*nobsy + inz*nobsz;

    Real Q = CsH0 * invr3;
    Real A = Q * 3 * rn;
    Real B = Q * CsH1;
    Real C = Q * CsH3;

    Real MT = Q*CsH2*sn + A*CsH1*sd;
    Real NT = B*sm + C*sd*rm;
    Real DT = invr*(B*3*sn*rm + C*sd*mn + A*(nu*sm - 5*sd*rm));
    Real ST = A*nu*rm + B*mn;

    sumx += nsrcx*NT + nobsx*MT + Dx*DT + ST*inx;
    sumy += nsrcy*NT + nobsy*MT + Dy*DT + ST*iny;
    sumz += nsrcz*NT + nobsz*MT + Dz*DT + ST*inz;
    ''',
    '''
    Real invr = rsqrt(r2);
    Real invr2 = invr * invr;
    Real invr3 = invr2 * invr;
    Real Dorx = invr * Dx;
    Real Dory = invr * Dy;
    Real Dorz = invr * Dz;

    Real rn = nsrcx * Dorx + nsrcy * Dory + nsrcz * Dorz;
    Real rm = nobsx * Dorx + nobsy * Dory + nobsz * Dorz;
    Real mn = nobsx * nsrcx + nobsy * nsrcy + nobsz * nsrcz;

    Real Q = CsH0 * invr3;
    Real A = Q * 3 * rn;
    Real B = Q * CsH1;
    Real C = Q * CsH3;

    Real MTx = Q*CsH2*nsrcx + A*CsH1*Dorx;
    Real MTy = Q*CsH2*nsrcy + A*CsH1*Dory;
    Real MTz = Q*CsH2*nsrcz + A*CsH1*Dorz;

    Real NTx = B*nobsx + C*Dorx*rm;
    Real NTy = B*nobsy + C*Dory*rm;
    Real NTz = B*nobsz + C*Dorz*rm;

    Real DTx = B*3*nsrcx*rm + C*Dorx*mn + A*(nu*nobsx - 5*Dorx*rm);
    Real DTy = B*3*nsrcy*rm + C*Dory*mn + A*(nu*nobsy - 5*Dory*rm);
    Real DTz = B*3*nsrcz*rm + C*Dorz*mn + A*(nu*nobsz - 5*Dorz*rm);

    Real ST = A*nu*rm + B*mn;

    Karr[0] = nsrcx*NTx + nobsx*MTx + Dorx*DTx + ST;
    Karr[1] = nsrcx*NTy + nobsx*MTy + Dorx*DTy;
    Karr[2] = nsrcx*NTz + nobsx*MTz + Dorx*DTz;
    Karr[3] = nsrcy*NTx + nobsy*MTx + Dory*DTx;
    Karr[4] = nsrcy*NTy + nobsy*MTy + Dory*DTy + ST;
    Karr[5] = nsrcy*NTz + nobsy*MTz + Dory*DTz;
    Karr[6] = nsrcz*NTx + nobsz*MTx + Dorz*DTx;
    Karr[7] = nsrcz*NTy + nobsz*MTy + Dorz*DTy;
    Karr[8] = nsrcz*NTz + nobsz*MTz + Dorz*DTz + ST;
    '''
)

elasticRT = Kernel(
    'elasticRT3', 3, 3, False, True, -2, 0, True,
    '''
    const Real nu = params[1];
    const Real CsRT0 = -1.0 / (8.0 * M_PI * (1.0 - nu));
    const Real CsRT1 = -(1 - 2 * nu) / (8.0 * M_PI * (1.0 - nu));
    const Real CsRT2 = -1.0 / (4 * M_PI);
    ''',
    '', '',
    surf_curl_obs = False,
    surf_curl_src = True,
    multipole_dim = 10
)

elasticRA = Kernel(
    'elasticRA3', 3, 3, True, False, -2, 0, True,
    '''
    const Real nu = params[1];
    const Real CsRT0 = -1.0 / (8.0 * M_PI * (1.0 - nu));
    const Real CsRT1 = -(1 - 2 * nu) / (8.0 * M_PI * (1.0 - nu));
    const Real CsRT2 = -1.0 / (4 * M_PI);
    ''',
    '', '',
    surf_curl_obs = True,
    surf_curl_src = False,
    multipole_dim = 7
)

elasticRH = Kernel(
    'elasticRH3', 3, 3, True, True, -1, 1, False,
    '''
    const Real G = params[0];
    const Real nu = params[1];
    const Real CsRH0 = G / (8.0 * M_PI * (1 - nu));
    const Real CsRH1 = 2 * (1 - nu);
    const Real CsRH2 = 4 * nu;
    ''',
    '', '',
    surf_curl_obs = True,
    surf_curl_src = True,
    multipole_dim = 22
)

laplace2S = Kernel(
    'laplaceS2', 2, 1, False, False, 'log', 0, False, '',
    '''''',
    '''
    Karr[0] = log(sqrt(r2)) / (2.0 * M_PI);
    '''
)

laplace2D = Kernel(
    'laplaceD2', 2, 1, False, True, -3, 0, False, '',
    '''''',
    '''
    Real r = sqrt(r2);
    Real rn = nsrcx * Dx + nsrcy * Dy;
    Karr[0] = rn / (2 * M_PI * r2);
    '''
)

laplace2H = Kernel(
    'laplaceH2', 2, 1, True, True, -2, 0, False, '',
    '''''',
    '''
    Real rn = nsrcx * Dx + nsrcy * Dy;
    Real rm = nobsx * Dx + nobsy * Dy;
    Real nm = nobsx * nsrcx + nobsy * nsrcy;
    Karr[0] = (-nm + ((2 * rn * rm) / r2)) / (2 * M_PI * r2);
    '''
)

laplace3S = Kernel(
    'laplaceS3', 3, 1, False, False, -3, 0, False,
    '''
    Real C = 1.0 / (4.0 * M_PI);
    ''',
    '''''',
    '''
    Karr[0] = rsqrt(r2) * C;
    '''
)

laplace3D = Kernel(
    'laplaceD3', 3, 1, False, True, -2, 0, False, '',
    '''''',
    '''
    Real r = sqrt(r2);
    Real rn = nsrcx * Dx + nsrcy * Dy + nsrcz * Dz;
    Karr[0] = rn / (4 * M_PI * r2 * r);
    '''
)

laplace3H = Kernel(
    'laplaceH3', 3, 1, True, True, -1, 0, False, '',
    '''''',
    '''
    Real r = sqrt(r2);
    Real rn = nsrcx * Dx + nsrcy * Dy + nsrcz * Dz;
    Real rm = nobsx * Dx + nobsy * Dy + nobsz * Dz;
    Real nm = nobsx * nsrcx + nobsy * nsrcy + nobsz * nsrcz;
    Karr[0] = - ((nm / (r * r2)) - ((3 * rn * rm)/(r2 * r2 * r))) / (4 * M_PI);
    '''
)

one2 = Kernel('one2', 2, 1, False, False, -4, 0, False, '', None, 'Karr[0] = 1.0;')
one3 = Kernel('one3', 3, 1, False, False, -4, 0, False, '', None, 'Karr[0] = 1.0;')
tensor_one3 = Kernel(
    'tensor_one3', 3, 3, False, False, -4, 0, False, '', None,
    '''
    Karr[0] = 1.0;
    Karr[1] = 0.0;
    Karr[2] = 0.0;
    Karr[3] = 0.0;
    Karr[4] = 0.0;
    Karr[5] = 0.0;
    Karr[6] = 0.0;
    Karr[7] = 0.0;
    Karr[8] = 0.0;
    '''
)
invr3 = Kernel(
    'invr3', 3, 3, False, False, -4, 0, False, '', None,
    '''
    Karr[0] = 1.0 / sqrt(r2);
    Karr[1] = 0.0;
    Karr[2] = 0.0;
    Karr[3] = 0.0;
    Karr[4] = 0.0;
    Karr[5] = 0.0;
    Karr[6] = 0.0;
    Karr[7] = 0.0;
    Karr[8] = 0.0;
    '''
)

def make_kernel_dict(ks):
    return {K.name: K for K in ks}

one_kernels = make_kernel_dict([one2, one3, tensor_one3])

laplace_kernels = make_kernel_dict([
    laplace2S, laplace2D, laplace2H,
    laplace3S, laplace3D, laplace3H
])

elastic_kernels = make_kernel_dict([
    elasticU,
    elasticT,
    elasticA,
    elasticH
])

regularized_elastic_kernels = make_kernel_dict([
    elasticRT,
    elasticRA,
    elasticRH,
])

etc_kernels = make_kernel_dict([invr3])

kernels = dict(one_kernels)
kernels.update(laplace_kernels)
kernels.update(elastic_kernels)
kernels.update(regularized_elastic_kernels)
kernels.update(etc_kernels)
