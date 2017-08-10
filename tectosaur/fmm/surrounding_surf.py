import numpy as np

def surrounding_surf_circle(order):
    pts = np.empty((order, 2))

    for i in range(order):
        theta = i * 2 * np.pi / order
        pts[i,0] = np.cos(theta)
        pts[i,1] = np.sin(theta)
    return pts

def surrounding_surf_sphere(order):
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
    return np.array(pts)

def surrounding_surf(order, dim):
    if dim == 2:
        return surrounding_surf_circle(order)
    else:
        return surrounding_surf_sphere(order)
