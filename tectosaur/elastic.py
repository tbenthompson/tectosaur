from tectosaur.tensors import vec_inner, tensor_outer, Ident, SYM, SKW
import sympy as sp

# The source for these kernels is Appendix 2 of Carini and Salvadori, 2002.
# The variables are given identical names

G = sp.symbols('G') # shear modulus
nu = sp.symbols('nu') # poisson ratio
b0, b1 = sp.symbols('b0, b1') # gravity kernel parameters

y = sp.symbols('yx, yy, yz') # The source point
x = sp.symbols('xx, xy, xz') # The observation point
l = sp.symbols('lx, ly, lz') # The source normal
n = sp.symbols('nx, ny, nz') # The observation normal

all_args = [G, nu, b0, b1]
all_args.extend(y)
all_args.extend(x)
all_args.extend(l)
all_args.extend(n)

d = [y[i] - x[i] for i in range(3)]

r = sp.sqrt(vec_inner(d, d))
d_o_d = tensor_outer(d, d)

# Carini and Salvadori appear to have switched the order of the outer products?
# At least, I can only match the auto-derived kernels when the outer product
# order is switched.
d_o_n = tensor_outer(n, d)
d_o_l = tensor_outer(l, d)
l_o_n = tensor_outer(n, l)

d_dot_n = vec_inner(d, n)
d_dot_l = vec_inner(d, l)
l_dot_n = vec_inner(l, n)

def U(i, j):
    d_o_n = tensor_outer(d, n)
    UC1 = 1 / (16 * sp.pi * G * (1 - nu))
    UC2 = 3 - 4 * nu
    return UC1 * (1 / r) * ((d_o_d[i][j] / (r ** 2)) + UC2 * Ident(i, j))

def T(i, j):
    AC1 = -1 / (8 * sp.pi * (1 - nu))
    AC2 = 1 - 2 * nu
    return AC1 * (1 / r ** 3) * (
        AC2 * (2 * SKW(d_o_l)[i][j] + d_dot_l * Ident(i, j)) +
        (3 * d_dot_l * d_o_d[i][j]) / (r ** 2)
    )

def A(i, j):
    AC1 = -1 / (8 * sp.pi * (1 - nu))
    AC2 = 1 - 2 * nu
    return AC1 * (1 / r ** 3) * (
        AC2 * (2 * SKW(d_o_n)[i][j] - d_dot_n * Ident(i, j)) -
        (3 * d_dot_n * d_o_d[i][j]) / (r ** 2)
    )

def H(i, j):
    HC1 = G * nu / (4 * sp.pi * (1 - nu))
    HC2 = (3 * nu - 1) / nu
    HC3 = (1 - nu) / nu
    HC4 = (1 - 2 * nu) / nu
    outer_factor = HC1 / r ** 3
    term1 = 2 * SYM(l_o_n)[i][j]
    term2 = 2 * SKW(l_o_n)[i][j] * HC2
    term3 = 3 * HC2 / (r ** 2) * (
        SKW(d_o_l)[i][j] * d_dot_n -
        SKW(d_o_n)[i][j] * d_dot_l
    )
    term4 = 3 * HC3 / (r ** 2) * (
        SYM(d_o_l)[i][j] * d_dot_n +
        SYM(d_o_n)[i][j] * d_dot_l
    )
    term5 = 3 * d_o_d[i][j] / (r ** 2) * (
        l_dot_n -
        (5 / nu) * (d_dot_n * d_dot_l) / (r ** 2)
    )
    term6 = Ident(i, j) * ((3 * d_dot_n * d_dot_l / r ** 2) + (l_dot_n * HC4))
    return outer_factor * (term1 + term2 + term3 + term4 + term5 + term6)

def main():
    import os
    import mako.template
    from tectosaur.sympy_to_cpp import to_cpp
    filenames = ['farfield.tcpp', 'farfield.tcu']
    out_filenames = []
    for f in filenames:
        t = mako.template.Template(filename = os.path.join('tectosaur', f))
        p = dict()
        p['kernel'] = [[to_cpp(H(i, j)) for j in range(3)] for i in range(3)]
        root, ext = os.path.splitext(f)
        out_file = root + '.' + ext[2:]
        open(os.path.join('tectosaur', out_file), 'w').write(t.render(**p))
        out_filenames.append(out_file)


    gitignore = '\n'.join(out_filenames + ['.gitignore'])
    open('tectosaur/.gitignore', 'w').write(gitignore)

if __name__ == '__main__':
    main()
