import numpy as np

def surrounding_surface_circle(order):
    pts = np.empty((order, 2))

    for i in range(order):
        theta = i * 2 * np.pi / order
        pts[i,0] = np.cos(theta)
        pts[i,1] = np.sin(theta)
    return pts

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
    return np.array(pts)

def surrounding_surface(order, dim):
    if dim == 2:
        return surrounding_surface_circle(order)
    else:
        return surrounding_surface_sphere(order)

def inscribe_surf(ball, scaling, surf):
    return surf * ball.R * scaling + ball.center

def c2e_solve(gpu_module, surf, bounds, check_r, equiv_r, K, params, float_type):
    import tectosaur.fmm.fmm_wrapper as fmm
    equiv_surf = inscribe_surf(bounds, equiv_r, surf)
    check_surf = inscribe_surf(bounds, check_r, surf)

    equiv_to_check = fmm.direct_matrix(
        gpu_module, K, check_surf, surf, equiv_surf, surf, params, float_type
    )

    out = np.linalg.pinv(equiv_to_check, rcond = 1e-15).flatten()
    return out

