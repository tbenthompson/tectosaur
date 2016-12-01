from tectosaur.standardize import *

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

def test_standardize():
    # out = standardize(np.array([[0,0,0],[0.2,0,0],[0.4,0.5,0]]))
    # np.testing.assert_almost_equal(out, [[0,0,0],[0.4,0.5,0],[0.2,0,0]])
    out,_,_,_,_ = standardize(np.array([[0,0,0],[1,0.0,0],[0.0,0.5,0]]), 20)
    np.testing.assert_almost_equal(out, [[0,0,0],[1.0,0.0,0],[0.2,0.4,0]])
