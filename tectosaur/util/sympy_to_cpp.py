import sympy
import mpmath

def cpp_binop(op):
    def _cpp_binop(expr, cache):
        out = '(' + to_cpp(expr.args[0], cache)
        for arg in expr.args[1:]:
            out += op + to_cpp(arg, cache)
        out += ')'
        return out
    return _cpp_binop

def cpp_func(f_name):
    def _cpp_func(expr, cache):
        return f_name + cpp_binop(',')(expr, cache)
    return _cpp_func

def cpp_pow(expr, cache):
    if expr.args[1] == -1:
        return '(1 / ' + to_cpp(expr.args[0], cache) + ')'
    elif expr.args[1] == 2:
        a = to_cpp(expr.args[0], cache)
        return '({a} * {a})'.format(a = a)
    elif expr.args[1] == -2:
        a = to_cpp(expr.args[0], cache)
        return '(1 / ({a} * {a}))'.format(a = a)
    elif expr.args[1] == -0.5:
        a = to_cpp(expr.args[0], cache)
        return '(1.0 / std::sqrt({a}))'.format(a = a)
    elif expr.args[1] == 0.5:
        a = to_cpp(expr.args[0], cache)
        return '(std::sqrt({a}))'.format(a = a)
    else:
        return cpp_func('std::pow')(expr, cache)

to_cpp_map = dict()
to_cpp_map[sympy.Mul] = cpp_binop('*')
to_cpp_map[sympy.Add] = cpp_binop('+')
to_cpp_map[sympy.Symbol] = lambda e, c: str(e)
to_cpp_map[sympy.Number] = lambda e, c: mpmath.nstr(float(e), 17)
to_cpp_map[sympy.numbers.Pi] = lambda e, c: 'M_PI'
to_cpp_map[sympy.NumberSymbol] = lambda e, c: str(e)
to_cpp_map[sympy.Function] = lambda e, c: cpp_func(str(e.func))(e, c)
to_cpp_map[sympy.Pow] = cpp_pow

def to_cpp(expr, cache, no_caching = False):
    if expr.args != () and not no_caching:
        if not str(expr) in cache:
            idx = cache.get('next_idx', 0)
            cache['next_idx'] = idx + 1
            cache[str(expr)] = ('tmp' + str(idx), to_cpp(expr, cache, no_caching = True), expr)
        return cache[str(expr)][0]
    if expr.func in to_cpp_map:
        return to_cpp_map[expr.func](expr, cache)
    for k in to_cpp_map:
        if issubclass(expr.func, k):
            return to_cpp_map[k](expr, cache)
