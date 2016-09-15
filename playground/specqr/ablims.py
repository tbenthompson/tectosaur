import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

angle_lim = 25

def lawcos(a, b, c):
    return np.arccos((a**2 + b**2 - c**2) / (2*a*b))

def shouldfilter(a, b):
    # filter out when L2 < 1
    L2 = np.sqrt((a-1)**2 + b**2)
    if L2 > 1:
        return True
    # filter out when L3 < 1
    L3 = np.sqrt(a**2 + b**2)
    if L3 > 1:
        return True

    # filter out when T1 < 20
    T1 = lawcos(1.0, L3, L2)
    if np.rad2deg(T1) < angle_lim:
        return True

    # filter out when T2 < 20
    T2 = lawcos(1.0, L2, L3)
    if np.rad2deg(T2) < angle_lim:
        return True

    # filter out when T3 < 20
    T3 = lawcos(L2, L3, 1.0)
    if np.rad2deg(T3) < angle_lim:
        return True
    return False

avs = []
bvs = []

for i in range(2000000):
    # choose random a,b where a is in [0, 1] and b is in [0, 1]
    params = np.random.rand(2)
    a = params[0] * 0.5
    b = params[1]
    if shouldfilter(a,b):
        continue

    avs.append(a)
    bvs.append(b)
minA = np.min(avs)
minB = np.min(bvs)
maxB = np.max(bvs)
print(minA, minB, maxB)

plt.plot(avs, bvs, '.')
plt.xlim([0,0.5])
plt.ylim([0,1])
plt.savefig('abrange.pdf')
import sys;sys.exit()
minlegalA = 0.100201758858;
minlegalB = 0.24132345693;
maxlegalB = 0.860758641203;


data = np.genfromtxt('ab.csv', delimiter=',')
Im = int(np.max(data[:,0]))
Jm = int(np.max(data[:,1]))
A = np.array(data[:, 2], dtype = np.float64)
B = np.array(data[:, 3], dtype = np.float64)
V = np.array(data[:, 4], dtype = np.float64)
Vim = V.reshape((Im+1,Jm+1))
Aim = A.reshape((Im+1,Jm+1))
Bim = B.reshape((Im+1,Jm+1))

cntf = plt.contourf(Aim, Bim, Vim, levels = np.linspace(np.min(V), np.max(V), 25))
plt.contour(
    Aim, Bim, Vim,
    levels = np.linspace(np.min(V), np.max(V), 25),
    linestyles = 'solid', colors = '#000000'
)
# plt.imshow(Vim)
plt.colorbar(cntf)
plt.show()

# for col in range(31):
#     skip = 2
#     subsetxs = Aim[::skip, col]
#     subsetvals = Vim[::skip, col]
#     xs = Aim[:, col]
#     vals = Vim[:, col]
#     vals_test = scipy.interpolate.barycentric_interpolate(subsetxs, subsetvals, xs)
#     diff = np.abs((vals_test - vals) / vals)
#     print(np.max(diff[:-1]))
#
# print("ROWS")
# print("ROWS")
# print("ROWS")
# start = 2
# skip = 2
# for row in range(16):
#     subsetxs = Bim[row, start::skip]
#     subsetvals = Vim[row, start::skip]
#     xs = Bim[row, start:]
#     vals = Vim[row, start:]
#     vals_test = scipy.interpolate.barycentric_interpolate(subsetxs, subsetvals, xs)
#     diff = np.abs((vals_test - vals) / vals)
#     print(np.max(diff[:]))

def interp(row, start, end, skip, loc):
    subsetxs = Bim[row, start:end:skip]
    subsetvals = Vim[row, start:end:skip]
    return scipy.interpolate.barycentric_interpolate(subsetxs, subsetvals, loc)

maxerr = 0.0
for row in range(16):
    for i in range(1000):
        params = np.random.rand(2)
        a = Aim[row, 0]
        b = minlegalB + (maxlegalB - minlegalB) * params[1]
        if shouldfilter(a,b):
            # print("SKIP",a,b)
            continue
        val1 = interp(row, 0, 30, 1, b)
        val2 = interp(row, 0, 30, 2, b)
        diff = val1 - val2
        relerr = np.abs(diff / val1)
        # print("GO",a,b,val1,val2,diff,relerr)
        maxerr = np.max([relerr,maxerr])
print("MAXERR!!!",maxerr)
