import numpy as np
import cppimport.import_hook
from tectosaur.nearfield.standardize import *
from tectosaur.util.geometry import tri_pt, linear_basis_tri
from tectosaur.util.test_decorators import slow, kernel

def test_origin_vertex():
    assert(get_origin_vertex(get_edge_lens([[0,0,0],[1,0,0],[0.2,0.5,0]])) == 0)
    assert(get_origin_vertex(get_edge_lens([[0,0,0],[1,0,0],[0.8,0.5,0]])) == 1)
    assert(get_origin_vertex(get_edge_lens([[1,0,0],[0,0,0],[0.2,0.5,0]])) == 1)
    assert(get_origin_vertex(get_edge_lens([[1,0,0],[0,0,0],[0.8,0.5,0]])) == 0)
    assert(get_origin_vertex(get_edge_lens([[0.8,0.5,0],[0,0,0],[1,0,0]])) == 2)

def test_translate():
    out,translation = translate([[0,1,0],[0,0,0],[0,2,0]])
    np.testing.assert_almost_equal(out, [[0,0,0], [0,-1,0],[0,1,0]])
    np.testing.assert_almost_equal(translation, [0,-1,0])

def test_relabel():
    out, labels = relabel([[0,0,0],[0.2,0,0],[0.4,0.5,0]], 0, 2)
    np.testing.assert_almost_equal(out, [[0,0,0],[0.4,0.5,0],[0.2,0,0]])
    assert(labels == [0,2,1])

def test_rotate1():
    out,R = rotate1_to_xaxis([[0,0,0], [1.0,1,1], [0,1,1]])
    np.testing.assert_almost_equal(out,
        [[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), -np.sqrt(1 / 3.0), -np.sqrt(1 / 3.0)]])

def test_rotate2():
    out1,R1 = rotate1_to_xaxis([[0,0,0], [1.0,1,1], [0,1,1]])
    out2,R2 = rotate2_to_xyplane(out1)
    np.testing.assert_almost_equal(out2,
        [[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), np.sqrt(2.0 / 3.0), 0]])

def test_rotate2_negative_y():
    tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    out, R2 = rotate2_to_xyplane(tri)
    np.testing.assert_almost_equal(R2, [[1,0,0], [0,0,1], [0,-1,0]])
    np.testing.assert_almost_equal(out, [[0,0,0], [2,0,0], [0,2,0]])

def test_scale():
    out,factor = scale([[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), np.sqrt(2.0 / 3.0), 0]])
    np.testing.assert_almost_equal(factor, 1.0 / np.sqrt(3))
    np.testing.assert_almost_equal(out, [[0,0,0],[1.0,0,0],[np.sqrt(4 / 9.0), np.sqrt(2 / 9.0),0]])

def test_check_bad_tri():
    assert(check_bad_tri([[0,0,0],[1,0,0],[0.1,0.01,0]], 20) == 3)
    assert(check_bad_tri([[0,0,0],[1,0,0],[20,20,0]], 20) == 1)
    assert(check_bad_tri([[0,0,0],[1,0,0],[0.5,0.5,0]], 20) == 0)

def test_standardize():
    code, out,_,_,_,_ = standardize([[0,0,0],[1,0.0,0],[0.0,0.5,0]], 20, True)
    assert(code == 0);
    np.testing.assert_almost_equal(out, [[0,0,0],[1.0,0.0,0],[0.2,0.4,0]])

def test_xz_vs_xy_plane():
    tri1 = [[1, -1,  0], [-1, -1, 0], [1, 1, 0]]
    tri2 = [[1, 0, -1], [-1, 0, -1], [1, 0, 1]]
    code, out, labels, trans, R, scale = standardize(tri1, 20, True)
    code2, out2, labels2, trans2, R2, scale2 = standardize(tri2, 20, True)
    np.testing.assert_almost_equal(out, out2)

# To test whether the standardization procedure and the transform_from_standard function
# are working properly, I compare the results of a direct integration with the results
# of a standardized triangle integration and subsequent unstandardization.
# There are some tricky/key pieces here. In particular, the starting epsilon value must be
# multiplied by (scale ** 2) in order to match the scaling of the triangle area.
# This is not strictly necessary, but avoiding it would require much higher accuracy and lower
# epsilon values before the scaled and unscaled integrals matched each other (in the limit,
# of course, they should match!)
def standardized_tri_tester(K, sm, pr, rho_order, theta_order, tol, starting_eps, n_eps, tri):
    include_log = True
    code, standard_tri, labels, translation, R, scale = standardize(
        np.array(tri), 20, True
    )
    is_flipped = not (labels[1] == ((labels[0] + 1) % 3))

    np.testing.assert_almost_equal(
        standard_tri,
        [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
    )

    import tectosaur_tables.coincident as coincident
    p = coincident.make_coincident_params(
        K, 1e-3, 25, True, True, 25, 25, starting_eps, n_eps, K == 'elasticH', 1, 1, 1
    )

    correct_limits = coincident.eval_tri_integral(tri, pr, p)
    correct_limits = correct_limits[:,0].reshape((3,3,3,3))

    # 1) calculate the standardized integrals
    p.starting_eps = starting_eps * scale
    epsvs, standard_vals = coincident.eval_tri_integral_no_limit(standard_tri, pr, p)

    # 2) convert them to the appropriate values for true triangles
    unstandardized_vals = np.array([
        np.array(transform_from_standard(
            # standard_vals[i,:], K, sm, labels, translation, R, scale
            standard_vals[i,:], K, sm, labels, translation, R, scale
        )).reshape(81)
        for i in range(standard_vals.shape[0])
    ])

    # 3) take the limit in true space, not standardized space
    unstandardized_limits = coincident.take_limits(
        epsvs / scale, unstandardized_vals,
        1 if p.include_log else 0, starting_eps
    )
    unstandardized_limits = unstandardized_limits[:,0].reshape((3,3,3,3))

    A = unstandardized_limits[0,0,0,0]
    B = correct_limits[0,0,0,0]

    # print(
    #     str(tol) +
    #     " " + str(starting_eps) +
    #     " " + str(n_eps) +
    #     " " + str(A) +
    #     " " + str(B)
    # )
    err = np.abs(
        (unstandardized_limits[:,0,:,0] - correct_limits[:,0,:,0]) /
        np.max(np.abs(correct_limits[:,0,:,0]))
    )
    assert(np.all(err < 0.03))
    # np.testing.assert_almost_equal(unstandardized_limits, correct_limits, 4)

def kernel_properties_tester(K, sm, pr):
    test_tris = [
        [[0.0,0.0,0.0], [1.0,0.0,0.0], [0.4,0.3,0.0]], #NO TRANSFORMATION
        [[0.0,0.0,0.0], [2.0,0.0,0.0], [0.8,0.6,0.0]], # JUST SCALE
        [[0.0,0.0,0.0], [0.0,1.0,0.0], [-0.3,0.4,0.0]], # JUST ROTATE
        [[0.0,0.0,0.0], [0.0,1.1,0.0], [-0.3,0.4,0.0]], # ROTATE + SCALE
        [[0.0,0.0,0.0], [0.0,0.0,1.1], [0.0,-0.3,0.4]], # ROTATE + SCALE
        [[0.0,0.0,0.0], [0.0,0.3,1.1], [0.0,-0.3,0.4]], # Two rotations and scalings
        [[0.0,0.0,0.0], [0.0,-0.3,0.4], [0.0,0.35,1.1]],
        [[0.0,0.35,1.1], [0.0,0.0,0.0], [0.0,-0.3,0.4]],
        [[0.0, -0.3, 0.4], [0.0,0.35,1.1], [0.0,0.0,0.0]],
        [[1.0,0.0,0.0], [0.0,-0.3,0.45], [0.0,0.35,1.1]]
    ]
    for t in test_tris:
        standardized_tri_tester(K, sm, pr, 50, 50, 0.005, 0.08, 3, t)

@slow
def test_kernel_transformation_properties(kernel):
    kernel_properties_tester(kernel, 1.0, 0.25)
