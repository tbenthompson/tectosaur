from tectosaur.standardize import *
import tectosaur.quadrature as quad
import tectosaur.limit as limit
from tectosaur_tables.gpu_integrator import coincident_integral, adjacent_integral
from tectosaur.test_decorators import slow


def test_origin_vertex():
    assert(get_origin_vertex(get_edge_lens(np.array([[0,0,0],[1,0,0],[0.2,0.5,0]]))) == 0)
    assert(get_origin_vertex(get_edge_lens(np.array([[0,0,0],[1,0,0],[0.8,0.5,0]]))) == 1)
    assert(get_origin_vertex(get_edge_lens(np.array([[1,0,0],[0,0,0],[0.2,0.5,0]]))) == 1)
    assert(get_origin_vertex(get_edge_lens(np.array([[1,0,0],[0,0,0],[0.8,0.5,0]]))) == 0)
    assert(get_origin_vertex(get_edge_lens(np.array([[0.8,0.5,0],[0,0,0],[1,0,0]]))) == 2)

def test_longest_edge():
    assert(get_longest_edge(get_edge_lens(np.array([[0,0,0],[1,0,0],[0.5,0.5,0]]))) == 0)
    assert(get_longest_edge(get_edge_lens(np.array([[0,0,0],[0.5,0.5,0],[1,0,0]]))) == 2)

def test_translate():
    out,translation = translate(np.array([[0,1,0],[0,0,0],[0,2,0]]))
    np.testing.assert_almost_equal(out, [[0,0,0], [0,-1,0],[0,1,0]])
    np.testing.assert_almost_equal(translation, [0,-1,0])

def test_relabel():
    out, labels = relabel(np.array([[0,0,0],[0.2,0,0],[0.4,0.5,0]]), 0, 2)
    np.testing.assert_almost_equal(out, [[0,0,0],[0.4,0.5,0],[0.2,0,0]])
    assert(labels == [0,2,1])

def test_rotate1():
    out,R = rotate1_to_xaxis(np.array([[0,0,0], [1.0,1,1], [0,1,1]], dtype = np.float64))
    np.testing.assert_almost_equal(out,
        [[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), -np.sqrt(1 / 3.0), -np.sqrt(1 / 3.0)]])

def test_rotate2():
    out1,R1 = rotate1_to_xaxis(np.array([[0,0,0], [1.0,1,1], [0,1,1]], dtype = np.float64))
    out2,R2 = rotate2_to_xyplane(out1)
    np.testing.assert_almost_equal(out2,
        [[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), np.sqrt(2.0 / 3.0), 0]])

def test_scale():
    out,factor = scale([[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), np.sqrt(2.0 / 3.0), 0]])
    np.testing.assert_almost_equal(factor, 1.0 / np.sqrt(3))
    np.testing.assert_almost_equal(out, [[0,0,0],[1.0,0,0],[np.sqrt(4 / 9.0), np.sqrt(2 / 9.0),0]])

def test_check_bad_tri():
    assert(check_bad_tri([[0,0,0],[1,0,0],[0.1,0.01,0]], 20) == 3)
    assert(check_bad_tri([[0,0,0],[1,0,0],[20,20,0]], 20) == 1)
    assert(check_bad_tri([[0,0,0],[1,0,0],[0.5,0.5,0]], 20) == 0)

def test_standardize():
    # out = standardize(np.array([[0,0,0],[0.2,0,0],[0.4,0.5,0]]))
    # np.testing.assert_almost_equal(out, [[0,0,0],[0.4,0.5,0],[0.2,0,0]])
    out,_,_,_,_ = standardize(np.array([[0,0,0],[1,0.0,0],[0.0,0.5,0]]), 20, True)
    np.testing.assert_almost_equal(out, [[0,0,0],[1.0,0.0,0],[0.2,0.4,0]])

def co_integrals(K, tri, rho_order, theta_order, tol, eps_start, n_steps, sm, pr):
    epsvs = eps_start * (2.0 ** (-np.arange(n_steps)))
    vals = []
    for eps in epsvs:
        vals.append(coincident_integral(
            tol, K, tri, eps, sm, pr, rho_order, theta_order
        ))
    vals = np.array(vals)
    return epsvs, vals

def adj_integrals(K, obs_tri, src_tri, rho_order, theta_order, tol, eps_start, n_steps, sm, pr):
    epsvs = eps_start * (2.0 ** (-np.arange(n_steps)))
    vals = []
    for eps in epsvs:
        vals.append(adjacent_integral(
            tol, K, obs_tri, src_tri, eps, sm, pr, rho_order, theta_order
        ))

    vals = np.array(vals)
    return epsvs, vals

def take_limits(epsvs, vals, include_log):
    return np.array([limit.limit(epsvs, vals[:, i], include_log) for i in range(81)])

# To test whether the standardization procedure and the transform_from_standard function
# are working properly, I compare the results of a direct integration with the results
# of a standardized triangle integration and subsequent unstandardization.
# There are some tricky/key pieces here. In particular, the starting epsilon value must be
# multiplied by (scale ** 2) in order to match the scaling of the triangle area.
# This is not strictly necessary, but avoiding it would require much higher accuracy and lower
# epsilon values before the scaled and unscaled integrals matched each other (in the limit,
# of course, they should match!)
def standardized_tri_tester(K, sm, pr, rho_order, theta_order, tol, eps_start, n_steps, tri):
    include_log = True
    standard_tri, labels, translation, R, scale = standardize(
        np.array(tri), 20, True
    )
    is_flipped = not (labels[1] == ((labels[0] + 1) % 3))

    np.testing.assert_almost_equal(
        standard_tri,
        [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
    )

    correct_epsvs, correct_vals = co_integrals(
        K, tri, rho_order, theta_order, tol, eps_start, n_steps, sm, pr
    )
    correct_limits = take_limits(correct_epsvs, correct_vals, include_log)
    correct_limits = correct_limits[:,0].reshape((3,3,3,3))

    # 1) calculate the standardized integrals
    epsvs, standard_vals = co_integrals(
        K, standard_tri, rho_order, theta_order, tol, eps_start * scale, n_steps, 1.0, pr
    )

    # 2) convert them to the appropriate values for true triangles
    unstandardized_vals = np.array([
        np.array(transform_from_standard(
            # standard_vals[i,:], K, sm, labels, translation, R, scale
            standard_vals[i,:], K, sm, labels, translation, R, scale
        )).reshape(81)
        for i in range(standard_vals.shape[0])
    ])

    # 3) take the limit in true space, not standardized space
    unstandardized_limits = take_limits(epsvs / scale, unstandardized_vals, include_log)
    unstandardized_limits = unstandardized_limits[:,0].reshape((3,3,3,3))

    A = unstandardized_limits[0,0,0,0]
    B = correct_limits[0,0,0,0]

    print(
        str(tol) +
        " " + str(eps_start) +
        " " + str(n_steps) +
        " " + str(A) +
        " " + str(B)
    )
    err = np.abs((unstandardized_limits[:,0,:,0] - correct_limits[:,0,:,0]) / np.max(np.abs(correct_limits[:,0,:,0])))
    assert(np.all(err < 0.03))
    # np.testing.assert_almost_equal(unstandardized_limits, correct_limits, 4)

def kernel_properties_tester(K, sm, pr):
    test_tris = [
        # [[0.0,0.0,0.0], [1.0,0.0,0.0], [0.4,0.3,0.0]], #NO TRANSFORMATION
        [[0.0,0.0,0.0], [2.0,0.0,0.0], [0.8,0.6,0.0]], # JUST SCALE
        [[0.0,0.0,0.0], [0.0,1.0,0.0], [-0.3,0.4,0.0]], # JUST ROTATE
        [[0.0,0.0,0.0], [0.0,1.1,0.0], [-0.3,0.4,0.0]], # ROTATE + SCALE
        [[0.0,0.0,0.0], [0.0,0.0,1.1], [0.0,-0.3,0.4]], # ROTATE + SCALE
        [[0.0,0.0,0.0], [0.0,0.3,1.1], [0.0,-0.3,0.4]],
        [[0.0,0.0,0.0], [0.0,-0.3,0.4], [0.0,0.35,1.1]],
        [[0.0,0.35,1.1], [0.0,0.0,0.0], [0.0,-0.3,0.4]],
        [[0.0, -0.3, 0.4], [0.0,0.35,1.1], [0.0,0.0,0.0]],
        [[1.0,0.0,0.0], [0.0,-0.3,0.45], [0.0,0.35,1.1]]
    ]
    for t in test_tris:
        print("TESTING " + str(t))
        standardized_tri_tester(K, sm, pr, 50, 50, 0.005, 0.08, 3, t)
        print("SUCCESS")

@slow
def test_U_properties():
    kernel_properties_tester('U', 1.0, 0.25)

@slow
def test_T_properties():
    kernel_properties_tester('T', 1.0, 0.25)

@slow
def test_A_properties():
    kernel_properties_tester('A', 1.0, 0.25)

@slow
def test_H_properties():
    kernel_properties_tester('H', 1.0, 0.25)
