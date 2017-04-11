import sympy
import mpmath


def cpp_binop(op):
    def _cpp_binop(expr):
        out = '((' + to_cpp(expr.args[0]) + ')'
        for arg in expr.args[1:]:
            out += op + '(' + to_cpp(arg) + ')'
        out += ')'
        return out
    return _cpp_binop

def cpp_func(f_name):
    def _cpp_func(expr):
        return f_name + cpp_binop(',')(expr)
    return _cpp_func

def cpp_pow(expr):
    if expr.args[1] == -1:
        return '(Real(1) / (' + to_cpp(expr.args[0]) + '))'
    elif expr.args[1] == 2:
        a = to_cpp(expr.args[0])
        return '(({a}) * ({a}))'.format(a = a)
    elif expr.args[1] == -2:
        a = to_cpp(expr.args[0])
        return '(Real(1) / (({a}) * ({a})))'.format(a = a)
    elif expr.args[1] == -0.5:
        a = to_cpp(expr.args[0])
        return '(Real(1) / sqrt({a}))'.format(a = a)
    elif expr.args[1] == -1.5:
        a = to_cpp(expr.args[0])
        return '(Real(1) / sqrt(({a}) * ({a}) * ({a})))'.format(a = a)
    elif expr.args[1] == 0.5:
        a = to_cpp(expr.args[0])
        return '(sqrt({a}))'.format(a = a)
    else:
        return cpp_func('pow')(expr)

to_cpp_map = dict()
to_cpp_map[sympy.Mul] = cpp_binop('*')
to_cpp_map[sympy.Add] = cpp_binop('+')
to_cpp_map[sympy.Symbol] = lambda ft, e: str(e)
to_cpp_map[sympy.Number] = lambda ft, e: ft + '(' + mpmath.nstr(float(e), 17) + ')'
to_cpp_map[sympy.numbers.Pi] = lambda ft, e: ft + '(M_PI)'
to_cpp_map[sympy.NumberSymbol] = lambda ft, e: ft + '(' + str(e) + ')'
to_cpp_map[sympy.Function] = lambda ft, e: cpp_func(str(e.func))(ft, e)
to_cpp_map[sympy.Pow] = cpp_pow

def to_cpp(expr, no_caching = False):
    mpmath.mp.dps = 50
    if expr.func in to_cpp_map:
        return to_cpp_map[expr.func](expr)
    for k in to_cpp_map:
        if issubclass(expr.func, k):
            return to_cpp_map[k](expr)
