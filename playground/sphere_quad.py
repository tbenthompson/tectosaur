import numpy as np
import sympy
import quadpy

def surrounding_surface_sphere(order):
    pts = []
    a = 4 * np.pi / order;
    d = np.sqrt(a);
    M_theta = int(np.round(np.pi / d))
    d_theta = np.pi / M_theta;
    d_phi = a / d_theta;
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta;
        M_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi;
            x = np.sin(theta) * np.cos(phi);
            y = np.sin(theta) * np.sin(phi);
            z = np.cos(theta);
            pts.append((x, y, z))
    return np.array(pts), np.ones(len(pts)) / len(pts)

def lebedev(order):
    q = quadpy.sphere.Lebedev(order)
    return q.points, q.weights

def integrate_exact(f, midpoint, radius):
    phi = sympy.Symbol('phi')
    theta = sympy.Symbol('theta')
    x_xi = np.array([[
        midpoint[0] + radius*sympy.sin(phi)*sympy.cos(theta),
        midpoint[1] + radius*sympy.sin(phi)*sympy.sin(theta),
        midpoint[2] + radius*sympy.cos(phi),
    ]])
    rtheta_x_rphi = sympy.sin(phi) * radius**2
    exact = sympy.integrate(
        sympy.integrate(rtheta_x_rphi * f(x_xi)[0], (phi, 0.0, sympy.pi)),
        (theta, 0, 2*sympy.pi)
    )
    return float(exact)

def test_lebedev():
    order = 5
    for i in range(2 * order):
        f = lambda x: x[:,0] ** i
        R = 1
        correct = integrate_exact(f, (0,0,0), R)

        q = lebedev(order)
        res = 4 * np.pi * R ** 2 * sum(f(q[0]) * q[1])
        print(res, correct)
        np.testing.assert_almost_equal(res, correct)

if __name__ == '__main__':
    test_lebedev()
