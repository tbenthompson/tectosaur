import numpy as np

def linear_basis_tri(xhat, yhat):
    return np.array([1.0 - xhat - yhat, xhat, yhat])

def tri_pt(basis, tri):
    return np.array([
        sum([basis[j] * tri[j][i] for j in range(3)])
        for i in range(3)
    ])

def cross(x, y):
    return np.array([
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0]
    ])

def tri_normal(tri, normalize = False):
    n = cross(
        [tri[2][i] - tri[0][i] for i in range(3)],
        [tri[2][i] - tri[1][i] for i in range(3)]
    )
    if normalize:
        n = n / np.linalg.norm(n)
    return n

class Side:
    front = 0
    behind = 1
    intersect = 2

def which_side_point(tri, pt):
    normal = tri_normal(tri)
    dot_val = (pt - tri[0]).dot(normal)
    if dot_val > 0:
        return Side.front
    elif dot_val < 0:
        return Side.behind
    else:
        return Side.intersect

def segment_side(sides):
    if sides[0] == sides[1]:
        return sides[0]
    elif sides[0] == Side.intersect:
        return sides[1]
    elif sides[1] == Side.intersect:
        return sides[0]
    else:
        return Side.intersect

def tri_side(s):
    edge0 = segment_side([s[0], s[1]]);
    edge1 = segment_side([s[0], s[2]]);
    edge2 = segment_side([s[1], s[2]]);
    if edge0 == Side.intersect and edge1 == edge2:
        return edge1;
    if edge1 == Side.intersect and edge2 == edge0:
        return edge2;
    if edge2 == Side.intersect and edge0 == edge1:
        return edge0;
    return edge0;

# template <size_t dim>
# Side which_side_point(const Vec<Vec<double,dim>,dim>& face,
#                 const Vec<double,dim>& pt)
# {
#     auto normal = unscaled_normal(face);
#     double dot_val = dot_product(pt - face[0], normal);
#     if (dot_val > 0) { return FRONT; }
#     else if (dot_val < 0) { return BEHIND; }
#     else { return INTERSECT; }
# }
#
# /* Returns the side of a plane that a triangle/segment is on. */
# template <size_t dim>
# Side facet_side(const std::array<Side,dim>& s);
#
# template <>
# inline Side facet_side<2>(const std::array<Side,2>& s)
# {
#     if (s[0] == s[1]) { return s[0]; }
#     else if (s[0] == INTERSECT) { return s[1]; }
#     else if (s[1] == INTERSECT) { return s[0]; }
#     else { return INTERSECT; }
# }
#
# template <>
# inline Side facet_side<3>(const std::array<Side,3>& s)
# {
#     auto edge0 = facet_side<2>({s[0], s[1]});
#     auto edge1 = facet_side<2>({s[0], s[2]});
#     auto edge2 = facet_side<2>({s[1], s[2]});
#     if (edge0 == INTERSECT && edge1 == edge2) {
#         return edge1;
#     }
#     if (edge1 == INTERSECT && edge2 == edge0) {
#         return edge2;
#     }
#     if (edge2 == INTERSECT && edge0 == edge1) {
#         return edge0;
#     }
#     return edge0;
# }
#
#
# /* Determine the side of the plane/line defined by triangle/segment
#  * that the given triangle/segment is on
#  */
# template <size_t dim>
# Side which_side_facet(const Vec<Vec<double,dim>,dim>& plane,
#     const Vec<Vec<double,dim>,dim>& face)
# {
#     std::array<Side,dim> sides;
#     for (size_t d = 0; d < dim; d++) {
#         sides[d] = which_side_point<dim>(plane, face[d]);
#     }
#     return facet_side<dim>(sides);
# }
