import tectosaur.mesh as mesh

def test_remove_duplicates():
    surface1 = mesh.rect_surface(2, 2, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    surface2 = mesh.rect_surface(2, 2, [[0, 0, 0], [-1, 0, 0], [-1, 1, 0], [0, 1, 0]])
    m_f = mesh.concat(surface1, surface2)
    assert(m_f[0].shape[0] == 6)
    assert(m_f[1].shape[0] == 4)
