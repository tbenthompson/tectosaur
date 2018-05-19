/*cppimport
<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'].extend([
    '../include/pybind11_nparray.hpp',
    '../include/vec_tensor.hpp',
    '../include/math_tools.hpp',
    'standardize.hpp',
])
cfg['parallel'] = False

from tectosaur.kernels import kernels
%>
*/

/*
 THE PLAN!
 Find longest edge
 relabel to put that longest edge as the 0-1 edge
 flip to put the 2 vertex closer to the 0 vertex than the 1 vertex
 translate 0 vertex to origin
 rotate 1 vertex to be at (A, 0, 0) and store rotation
 rotate 2 vertex to be at (B, C, 0) with C > 0 and store rotation
 scale triangle so that 1 vertex is at (1, 0, 0) and store scale factor
 check that triangle internal angles are greater than 20 degrees
*/

#include "include/math_tools.hpp"
#include "standardize.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;
//TODO: Can the pyshim functions be written with fancy C++14 templating and such?

int get_origin_vertex(const Vec3& lens) {
    auto longest = get_longest_edge(lens);
    if (longest == 0 && lens[1] >= lens[2]) {
        return 0;
    } else if (longest == 0 && lens[2] >= lens[1]) {
        return 1;
    } else if (longest == 1 && lens[2] >= lens[0]) {
        return 1;
    } else if (longest == 1 && lens[0] >= lens[2]) {
        return 2;
    } else if (longest == 2 && lens[0] >= lens[1]) {
        return 2;
    } else {//if (longest == 2 && lens[1] >= lens[0]) {
        return 0;
    }
}

std::pair<Tensor3,std::array<int,3>> relabel(
    const Tensor3& tri, int ov, int longest_edge) 
{
    std::array<int,3> labels;

    if (longest_edge == ov) {
        labels = {ov, (ov + 1) % 3, (ov + 2) % 3};
    } else if ((longest_edge + 1) % 3 == ov) {
        labels = {ov, (ov + 2) % 3, (ov + 1) % 3};
    } else {
        throw std::runtime_error("BAD!");
    }
    Tensor3 out;
    for (int i = 0; i < 3; i++) {
        out[i] = tri[labels[i]];
    }
    return {out, labels};
}

py::tuple relabel_shim(const Tensor3& tri, int ov, int longest_edge) {
    auto out = relabel(tri, ov, longest_edge);
    return py::make_tuple(out.first, out.second);
}

int check_bad_tri(const Tensor3& tri, double angle_lim) {
    double eps = 1e-10;

    auto a = tri[2][0];
    auto b = tri[2][1];

    // filter out when L2 > 1
    auto L2 = sqrt(std::pow(a-1,2) + b*b);
    if (L2 > 1 + eps) {
        return 1;
    }

    // filter out when L3 > 1
    auto L3 = sqrt(a*a + b*b);
    if (L3 >= 1 + eps) {
        return 2;
    }

    // filter out when T1 < angle_lim
    auto A1 = lawcos(1.0, L3, L2);
    if (rad2deg(A1) < angle_lim - eps) {
        return 3;
    }

    // filter out when A2 < angle_lim
    auto A2 = lawcos(1.0, L2, L3);
    if (rad2deg(A2) < angle_lim - eps) {
        return 4;
    }

    // filter out when A3 < angle_lim
    auto A3 = lawcos(L2, L3, 1.0);
    if (rad2deg(A3) < angle_lim - eps) {
        return 5;
    }

    return 0;
}

std::pair<Tensor3,Vec3> translate(const Tensor3& tri) {
    Vec3 translation; 
    Tensor3 out_tri;
    for (int d1 = 0; d1 < 3; d1++) {
        translation[d1] = -tri[0][d1];
    }
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out_tri[d1][d2] = tri[d1][d2] + translation[d2];
        }
    }
    return {out_tri, translation}; 
}

py::tuple translate_pyshim(const Tensor3& tri) {
    auto out = translate(tri);
    return py::make_tuple(out.first, out.second);
}

std::pair<Tensor3,Tensor3> rotate1_to_xaxis(const Tensor3& tri) {
    // rotate 180deg around the axis halfway in between the 0-1 vector and
    // the 0-(L,0,0) vector, where L is the length of the 0-1 vector
    auto to_pt1 = sub(tri[1], tri[0]);
    auto pt1L = length(to_pt1);
    Vec3 to_target = {pt1L, 0, 0};
    auto axis = div(add(to_pt1, to_target), 2.0);
    auto axis_mag = length(axis); 
    if (axis_mag == 0.0) {
        axis = {0, 0, 1};
    } else {
        axis = div(axis, axis_mag);
    }
    double theta = M_PI;
    auto rot_mat = rotation_matrix(axis, theta);
    auto out_tri = transpose(mult(rot_mat, transpose(tri)));
    return {out_tri, rot_mat};
}

py::tuple rotate1_to_xaxis_pyshim(const Tensor3& tri) {
    auto out = rotate1_to_xaxis(tri);
    return py::make_tuple(out.first, out.second);
}

std::pair<Tensor3,Tensor3> rotate2_to_xyplane(const Tensor3& tri) {
    Vec3 xaxis = {1, 0, 0};

    // Find angle to rotate to reach x-y plane
    double ydot2 = tri[2][1] / length({0, tri[2][1], tri[2][2]});
    double theta = acos(ydot2);

    auto rot_mat = rotation_matrix(xaxis, theta);
    auto out_tri = transpose(mult(rot_mat, transpose(tri)));

    if (fabs(out_tri[2][2]) > 1e-10) {
        theta = -acos(ydot2);
        rot_mat = rotation_matrix(xaxis, theta);
        out_tri = transpose(mult(rot_mat, transpose(tri)));
    }

    if (out_tri[2][1] < 0) {
        Tensor3 flip_y_mat = rotation_matrix(xaxis, M_PI);
        out_tri = transpose(mult(flip_y_mat, transpose(out_tri)));
        rot_mat = mult(flip_y_mat, rot_mat);
    }
    return {out_tri, rot_mat};
}

py::tuple rotate2_to_xyplane_pyshim(const Tensor3& tri) {
    auto out = rotate2_to_xyplane(tri);
    return py::make_tuple(out.first, out.second);
}

std::pair<Tensor3,Tensor3> full_standardize_rotate(const Tensor3& tri) {
    auto rot1 = rotate1_to_xaxis(tri);
    auto rot2 = rotate2_to_xyplane(rot1.first);
    return {rot2.first, mult(rot2.second, rot1.second)};
}

py::tuple full_standardize_rotate_pyshim(const Tensor3& tri) {
    auto out = full_standardize_rotate(tri);
    return py::make_tuple(out.first, out.second);
}

std::pair<Tensor3,double> scale(const Tensor3& tri) {
    double scale = 1.0 / tri[1][0];
    Tensor3 out_tri;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out_tri[d1][d2] = tri[d1][d2] * scale;
        }
    }
    return {out_tri, scale}; 
}

py::tuple scale_pyshim(const Tensor3& tri) {
    auto out = scale(tri);
    return py::make_tuple(out.first, out.second);
}

StandardizeResult standardize(const Tensor3& tri, double angle_lim, bool should_relabel) {
    std::array<int,3> labels;
    Tensor3 relabeled;
    if (should_relabel) {
        auto ls = get_edge_lens(tri);
        auto longest = get_longest_edge(ls);
        auto ov = get_origin_vertex(ls);
        auto relabel_out = relabel(tri, ov, longest);
        relabeled = relabel_out.first;
        labels = relabel_out.second;
    } else {
        relabeled = tri;
        labels = {0,1,2};
    }
    auto trans_out = translate(relabeled);
    auto rot_out = full_standardize_rotate(trans_out.first);
    auto scale_out = scale(rot_out.first);
    int code = check_bad_tri(scale_out.first, angle_lim);
    if (should_relabel && code != 0) {
        throw BadTriangleException(std::to_string(code));
    }
    return {
        code, scale_out.first, labels, trans_out.second, rot_out.second, scale_out.second
    };
}

py::tuple standardize_pyshim(const Tensor3& tri, double angle_lim, bool should_relabel) {
    auto out = standardize(tri, angle_lim, should_relabel);
    return py::make_tuple(
        out.bad_tri_code, out.tri, out.labels, out.translation, out.R, out.scale
    );
}

std::array<double,81> transform_from_standard(const std::array<double,81>& I,
    const KernelProps& k_props, double sm, const std::array<int,3>& labels,
    const Vec3& translation, const Tensor3& R, double scale) 
{
    std::array<double,81> out;

    bool flip_negate = (labels[1] != (labels[0] + 1) % 3) && k_props.flip_negate;
    double scale_times_sm = std::pow(scale, k_props.scale_power) * std::pow(sm, k_props.sm_power);

    for (int sb1 = 0; sb1 < 3; sb1++) {
        for (int sb2 = 0; sb2 < 3; sb2++) {
            auto cb1 = labels[sb1];
            auto cb2 = labels[sb2];
            
            Tensor3 I_chunk;
            for (int d1 = 0; d1 < 3; d1++) {
                for (int d2 = 0; d2 < 3; d2++) {
                    I_chunk[d1][d2] = I[sb1 * 27 + d1 * 9 + sb2 * 3 + d2];
                }
            }

            // Multiply by rotation tensor.
            Tensor3 I_rot = mult(mult(transpose(R), I_chunk), R);

            // Multiply by scale
            // Multiply by shear mod.
            Tensor3 I_scale = mult(I_rot, scale_times_sm);

            // Output
            for (int d1 = 0; d1 < 3; d1++) {
                for (int d2 = 0; d2 < 3; d2++) {
                    if (flip_negate && d1 != d2) {
                    // if (flip_negate) {
                        out[cb1 * 27 + d1 * 9 + cb2 * 3 + d2] = -I_scale[d1][d2];
                    } else {
                        out[cb1 * 27 + d1 * 9 + cb2 * 3 + d2] = I_scale[d1][d2];
                    }
                }
            }
        }
    }
    return out;
}

KernelProps get_kernel_props(std::string K) {
    % for k_name, K in kernels.items():
        % if K.spatial_dim == 3:
            if (K == "${K.name}") { 
                return {
                    ${K.scale_type}, ${K.sm_power}, ${str(K.flip_negate).lower()}
                };
            }
        % endif
    % endfor
    else { throw std::runtime_error("unknown kernel"); }
}

std::array<double,81> transform_from_standard_pyshim(const std::array<double,81>& I,
    std::string K, double sm, const std::array<int,3>& labels,
    const Vec3& translation, const Tensor3& R, double scale) {

    return transform_from_standard(I, get_kernel_props(K), sm, labels, translation, R, scale);
}

PYBIND11_MAKE_OPAQUE(std::vector<StandardizeResult>);
PYBIND11_MODULE(standardize, m) {
    m.def("get_edge_lens", get_edge_lens);
    m.def("get_longest_edge", get_longest_edge);
    m.def("get_origin_vertex", get_origin_vertex);
    m.def("relabel", relabel);
    m.def("check_bad_tri", check_bad_tri);
    m.def("translate", translate_pyshim);
    m.def("rotate1_to_xaxis", rotate1_to_xaxis_pyshim);
    m.def("rotate2_to_xyplane", rotate2_to_xyplane_pyshim);
    m.def("full_standardize_rotate", full_standardize_rotate_pyshim);
    m.def("scale", scale_pyshim);
    py::class_<StandardizeResult>(m, "StandardizeResult");
    py::class_<std::vector<StandardizeResult>>(m, "ListStandardizeResult");
    m.def("standardize", standardize_pyshim);
    m.def("transform_from_standard", transform_from_standard_pyshim);
    py::register_exception<BadTriangleException>(m, "BadTriangleException");
}
