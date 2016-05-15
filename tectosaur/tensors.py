def tensor_outer(a, b):
    dim = len(a)
    return [[a[i] * b[j] for j in range(dim)] for i in range(dim)]

def tensor_sum(a, b):
    dim = len(a)
    return [[a[i][j] + b[i][j] for j in range(dim)] for i in range(dim)]

def tensor_negate(a):
    dim = len(a)
    return [[-a[i][j] for j in range(dim)] for i in range(dim)]

def tensor_mult(a, factor):
    dim = len(a)
    return [[a[i][j] * factor for j in range(dim)] for i in range(dim)]

def transpose(t):
    dim = len(t)
    return [[t[j][i] for j in range(dim)] for i in range(dim)]

def SKW(t):
    return tensor_mult(tensor_sum(tensor_negate(transpose(t)), t), 0.5)

def SYM(t):
    return tensor_mult(tensor_sum(transpose(t), t), 0.5)

def Ident(i, j):
    if i == j:
        return 1
    return 0
