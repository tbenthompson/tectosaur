class Kernel:
    def __init__(self, name, needs_obsn, needs_srcn, scale_type, sm_power, flip_negate,
            constants_code, vector_code, tensor_code):
        self.name = name
        self.needs_obsn = needs_obsn
        self.needs_srcn = needs_srcn
        self.scale_type = scale_type
        self.sm_power = sm_power
        self.flip_negate = flip_negate
        self.constants_code = constants_code
        self.vector_code = vector_code
        self.tensor_code = tensor_code

elasticU = Kernel(
    'elasticU', False, False, -3, 1, False,
    '''
    const Real G = params[0];
    const Real nu = params[1];
    const Real CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
    const Real CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
    ''',
    '''
    Real invr = 1.0 / sqrt(r2);
    Real Q1 = CsU0 * invr;
    Real Q2 = CsU1 * invr / r2;
    Real ddi = Dx*inx + Dy*iny + Dz*inz;
    sumx += Q1*inx + Q2*Dx*ddi;
    sumy += Q1*iny + Q2*Dy*ddi;
    sumz += Q1*inz + Q2*Dz*ddi;
    ''',
    '''
    Real invr = 1.0 / sqrt(r2);
    Real Q1 = CsU0 * invr;
    Real Q2 = CsU1 * invr / r2;
    Real K00 = Q2*Dx*Dx + Q1;
    Real K01 = Q2*Dx*Dy;
    Real K02 = Q2*Dx*Dz;
    Real K10 = Q2*Dy*Dx;
    Real K11 = Q2*Dy*Dy + Q1;
    Real K12 = Q2*Dy*Dz;
    Real K20 = Q2*Dz*Dx;
    Real K21 = Q2*Dz*Dy;
    Real K22 = Q2*Dz*Dz + Q1;
    '''
)

TA_const_code = '''
    const Real G = params[0];
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
        Real invr = 1.0 / sqrt(r2);
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
        Real invr = 1.0 / sqrt(r2);
        Real invr2 = invr * invr;
        Real invr3 = invr2 * invr;

        Real rn = {n_name}x * Dx + {n_name}y * Dy + {n_name}z * Dz;

        Real A = {plus_or_minus}CsT0 * invr3;
        Real C = {minus_or_plus}CsT1 * invr3 * invr2;

        Real nxdy = {n_name}x*Dy-{n_name}y*Dx;
        Real nzdx = {n_name}z*Dx-{n_name}x*Dz;
        Real nzdy = {n_name}z*Dy-{n_name}y*Dz;

        Real K00 = A * -rn                  + C*Dx*rn*Dx;
        Real K01 = A * {minus_or_plus}nxdy + C*Dx*rn*Dy;
        Real K02 = A * {plus_or_minus}nzdx + C*Dx*rn*Dz;
        Real K10 = A * {plus_or_minus}nxdy + C*Dy*rn*Dx;
        Real K11 = A * -rn                  + C*Dy*rn*Dy;
        Real K12 = A * {plus_or_minus}nzdy + C*Dy*rn*Dz;
        Real K20 = A * {minus_or_plus}nzdx + C*Dz*rn*Dx;
        Real K21 = A * {minus_or_plus}nzdy + C*Dz*rn*Dy;
        Real K22 = A * -rn                  + C*Dz*rn*Dz;
    '''.format(**TA_args(k_name))

elasticT = Kernel(
    'elasticT', False, True, -2, 0, True,
    TA_const_code, TA_vector_code('T'), TA_tensor_code('T')
)

elasticA = Kernel(
    'elasticA', True, False, -2, 0, True,
    TA_const_code, TA_vector_code('A'), TA_tensor_code('A')
)

elasticH = Kernel(
    'elasticH', True, True, -1, -1, False,
    '''
    const Real G = params[0];
    const Real nu = params[1];
    const Real CsH0 = G/(4*M_PI*(1-nu));
    const Real CsH1 = 1-2*nu;
    const Real CsH2 = -1+4*nu;
    const Real CsH3 = 3*nu;
    ''',
    '''
    Real invr = 1.0 / sqrt(r2);
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
    Real invr = 1.0 / sqrt(r2);
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

    Real K00 = nsrcx*NTx + nobsx*MTx + Dorx*DTx + ST;
    Real K01 = nsrcx*NTy + nobsx*MTy + Dorx*DTy;
    Real K02 = nsrcx*NTz + nobsx*MTz + Dorx*DTz;
    Real K10 = nsrcy*NTx + nobsy*MTx + Dory*DTx;
    Real K11 = nsrcy*NTy + nobsy*MTy + Dory*DTy + ST;
    Real K12 = nsrcy*NTz + nobsy*MTz + Dory*DTz;
    Real K20 = nsrcz*NTx + nobsz*MTx + Dorz*DTx;
    Real K21 = nsrcz*NTy + nobsz*MTy + Dorz*DTy;
    Real K22 = nsrcz*NTz + nobsz*MTz + Dorz*DTz + ST;
    '''
)

kernels = [
    elasticU,
    elasticT,
    elasticA,
    elasticH
]

# % if k_name is 'invr':
#     Real kernel_val = 1.0 / sqrt(r2);
#     sumx += kernel_val * inx;
# % elif k_name is 'tensor_invr':
#     Real kernel_val = 1.0 / sqrt(r2);
#     Real insum = inx + iny + inz;
#     sumx += kernel_val * insum;
#     sumy += kernel_val * insum;
#     sumz += kernel_val * insum;
# % elif k_name is 'one':
#     sumx += inx;
# % elif k_name is 'laplace_double':
#     Real r = sqrt(r2);
#     Real rn = nsrcx * Dx + nsrcy * Dy + nsrcz * Dz;
#     Real kernel_val = rn / (4 * M_PI * r2 * r);
#     sumx += kernel_val * inx;
