from math import factorial
import scipy.special
import numpy as np

def sloppy_spherical(y):
    r = np.linalg.norm(y)
    costheta = y[2] / r
    theta = np.arccos(costheta)
    phi = np.arccos(y[0] / r / np.sin(theta))
    return r, theta, phi

def Rdirect(n_max, y):
    r, theta, phi = sloppy_spherical(y)
    real = np.zeros((n_max + 1, 2 * n_max + 1))
    imag = np.zeros((n_max + 1, 2 * n_max + 1))
    Pmn = scipy.special.lpmn(n_max, n_max, np.cos(theta))[0]
    for i in range(n_max + 1):
        for j in range(-i, i + 1):
            if j < 0:
                lp = (
                    ((-1) ** (-j)) * (factorial(i + j) / factorial(i - j))
                    * Pmn[-j, i] / ((-1) ** -j)
                )
            else:
                lp = Pmn[j, i] / ((-1) ** j)
            factor = (r ** i) * lp / factorial(i + j)
            real[i, n_max + j] = factor * np.cos(j * phi)
            imag[i, n_max + j] = factor * np.sin(j * phi)
    return real, imag

def Sdirect(n_max, y):
    r, theta, phi = sloppy_spherical(y)
    real = np.zeros((n_max + 1, 2 * n_max + 1))
    imag = np.zeros((n_max + 1, 2 * n_max + 1))
    Pmn = scipy.special.lpmn(n_max, n_max, np.cos(theta))[0]
    for i in range(n_max + 1):
        for j in range(-i, i + 1):
            if j < 0:
                lp = (
                    ((-1) ** (-j)) * (factorial(i + j) / factorial(i - j))
                    * Pmn[-j, i] / ((-1) ** -j)
                )
            else:
                lp = Pmn[j, i] / ((-1) ** j)
            factor = factorial(i - j) * lp / (r ** (i + 1))
            real[i, n_max + j] = factor * np.cos(j * phi)
            imag[i, n_max + j] = factor * np.sin(j * phi)
    return real, imag

def R(n_max, y):
    y1, y2, y3 = y
    real = np.zeros((n_max + 1, 2 * n_max + 1))
    imag = np.zeros((n_max + 1, 2 * n_max + 1))
    real[0, n_max] = 1.0
    for i in range(0, n_max):
        real[i + 1, n_max + i + 1] = (
            (y1 * real[i, n_max + i] - y2 * imag[i, n_max + i])
            / (2 * (i + 1))
        )
        imag[i + 1, n_max + i + 1] = (
            (y1 * imag[i, n_max + i] + y2 * real[i, n_max + i])
            / (2 * (i + 1))
        )

    t2f = np.linalg.norm(y) ** 2
    for j in range(n_max + 1):
        for i in range(j, n_max):
            factor = 1.0 / ((i + 1) ** 2 - j ** 2)
            t1f = (2 * i + 1) * y3
            real[i + 1, n_max + j] = factor * (t1f * real[i, n_max + j] - t2f * real[i - 1, n_max + j])
            imag[i + 1, n_max + j] = factor * (t1f * imag[i, n_max + j] - t2f * imag[i - 1, n_max + j])
    for i in range(n_max + 1):
        for j in range(1, n_max + 1):
            real[i, n_max - j] = ((-1) ** j) * real[i, n_max + j]
            imag[i, n_max - j] = ((-1) ** (j + 1)) * imag[i, n_max + j]
    return real, imag

def R_storagefree(n_max, y):
    def neg(real, imag, mi):
        return (
            ((-1) ** mi) * real,
            ((-1) ** (mi + 1)) * imag
        )

    y1, y2, y3 = y
    real = np.zeros((n_max + 1, 2 * n_max + 1))
    imag = np.zeros((n_max + 1, 2 * n_max + 1))

    t2f = np.linalg.norm(y) ** 2
    Rsr = 1.0
    Rsi = 0.0
    for mi in range(0, n_max + 1):
        real[mi, n_max + mi] = Rsr
        imag[mi, n_max + mi] = Rsi
        real[mi, n_max - mi], imag[mi, n_max - mi] = neg(Rsr, Rsi, mi)

        Rm2r = 0.0
        Rm2i = 0.0
        Rm1r = Rsr
        Rm1i = Rsi
        for ni in range(mi, n_max):
            factor = 1.0 / ((ni + 1) ** 2 - mi ** 2)
            t1f = (2 * ni + 1) * y3
            Rvr = factor * (t1f * Rm1r - t2f * Rm2r)
            Rvi = factor * (t1f * Rm1i - t2f * Rm2i)
            real[ni + 1, n_max + mi] = Rvr
            imag[ni + 1, n_max + mi] = Rvi
            real[ni + 1, n_max - mi], imag[ni + 1, n_max - mi] = neg(Rvr, Rvi, mi)
            Rm2r = Rm1r
            Rm2i = Rm1i
            Rm1r = Rvr
            Rm1i = Rvi
        Rsrold = Rsr
        Rsiold = Rsi
        Rsr = (y1 * Rsrold - y2 * Rsiold) / (2 * (mi + 1))
        Rsi = (y1 * Rsiold + y2 * Rsrold) / (2 * (mi + 1))
    return real, imag


def S(n_max, y):
    y1, y2, y3 = y
    ynorm = np.linalg.norm(y)
    ynorm2 = ynorm ** 2
    real = np.zeros((n_max + 1, 2 * n_max + 1))
    imag = np.zeros((n_max + 1, 2 * n_max + 1))
    real[0, n_max] = 1.0 / ynorm
    for i in range(0, n_max):
        factor = (2 * i + 1) / ynorm2
        real[i + 1, n_max + i + 1] = factor * (
            (y1 * real[i, n_max + i] - y2 * imag[i, n_max + i])
        )
        imag[i + 1, n_max + i + 1] = factor * (
            (y1 * imag[i, n_max + i] + y2 * real[i, n_max + i])
        )

    for j in range(n_max + 1):
        for i in range(j, n_max):
            factor = 1.0 / ynorm2
            t1f = (2 * i + 1) * y3
            t2f = i ** 2 - j ** 2
            real[i + 1, n_max + j] = factor * (
                t1f * real[i, n_max + j] - t2f * real[i - 1, n_max + j]
            )
            imag[i + 1, n_max + j] = factor * (
                t1f * imag[i, n_max + j] - t2f * imag[i - 1, n_max + j]
            )
    for i in range(n_max + 1):
        for j in range(1, n_max + 1):
            real[i, n_max - j] = ((-1) ** j) * real[i, n_max + j]
            imag[i, n_max - j] = ((-1) ** (j + 1)) * imag[i, n_max + j]
    return real, imag

def S_storagefree(n_max, y):
    def neg(real, imag, mi):
        return (
            ((-1) ** mi) * real,
            ((-1) ** (mi + 1)) * imag
        )

    y1, y2, y3 = y
    real = np.zeros((n_max + 1, 2 * n_max + 1))
    imag = np.zeros((n_max + 1, 2 * n_max + 1))

    ynorm = np.linalg.norm(y)
    ynorm2 = ynorm ** 2
    Ssr = 1.0 / ynorm
    Ssi = 0.0
    for mi in range(0, n_max + 1):
        real[mi, n_max + mi] = Ssr
        imag[mi, n_max + mi] = Ssi
        real[mi, n_max - mi], imag[mi, n_max - mi] = neg(Ssr, Ssi, mi)

        Sm2r = 0.0
        Sm2i = 0.0
        Sm1r = Ssr
        Sm1i = Ssi
        for ni in range(mi, n_max):
            factor = 1.0 / ynorm2
            t1f = (2 * ni + 1) * y3
            t2f = ni ** 2 - mi ** 2
            Svr = factor * (t1f * Sm1r - t2f * Sm2r)
            Svi = factor * (t1f * Sm1i - t2f * Sm2i)
            real[ni + 1, n_max + mi] = Svr
            imag[ni + 1, n_max + mi] = Svi
            real[ni + 1, n_max - mi], imag[ni + 1, n_max - mi] = neg(Svr, Svi, mi)
            Sm2r = Sm1r
            Sm2i = Sm1i
            Sm1r = Svr
            Sm1i = Svi
        Ssrold = Ssr
        Ssiold = Ssi
        factor = (2 * mi + 1) / ynorm2
        Ssr = factor * (y1 * Ssrold - y2 * Ssiold)
        Ssi = factor * (y1 * Ssiold + y2 * Ssrold)
    return real, imag

def Sderivs(n_max, y, d):
    Svr, Svi = S(n_max + 1, y)
    real = np.zeros((n_max + 1, 2 * n_max + 1))
    imag = np.zeros((n_max + 1, 2 * n_max + 1))

    if d == 0:
        for i in range(n_max + 1):
            for j in range(-i, i + 1):
                real[i, n_max + j] = 0.5 * (
                    Svr[i + 1, (n_max + 1) + j - 1]
                    - Svr[i + 1, (n_max + 1) + j + 1]
                )
                imag[i, n_max + j] = 0.5 * (
                    Svi[i + 1, (n_max + 1) + j - 1]
                    - Svi[i + 1, (n_max + 1) + j + 1]
                )
    elif d == 1:
        for i in range(n_max + 1):
            for j in range(-i, i + 1):
                real[i, n_max + j] = -0.5 * (
                    Svi[i + 1, (n_max + 1) + j - 1]
                    + Svi[i + 1, (n_max + 1) + j + 1]
                )
                imag[i, n_max + j] = 0.5 * (
                    Svr[i + 1, (n_max + 1) + j - 1]
                    + Svr[i + 1, (n_max + 1) + j + 1]
                )
    else:
        for i in range(n_max + 1):
            for j in range(-i, i + 1):
                real[i, n_max + j] = -Svr[i + 1, (n_max + 1) + j]
                imag[i, n_max + j] = -Svi[i + 1, (n_max + 1) + j]
    return real, imag

