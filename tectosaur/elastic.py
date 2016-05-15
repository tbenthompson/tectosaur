import tectosaur.tensors
import sympy as sp

shear_mod = sp.symbols('G')
poisson = sp.symbols('nu')

def U():
    return (1 / (16 * sp.pi)) * (1 / (G * (1 - nu))) * (1 / r)
