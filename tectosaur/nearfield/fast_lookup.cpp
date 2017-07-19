<%
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-Wall', '-fopenmp'])
cfg['linker_args'] = ['-fopenmp']
cfg['dependencies'] = [
    '../include/pybind11_nparray.hpp',
    '../include/vec_tensor.hpp',
    '../include/timing.hpp'
]
from tectosaur.nearfield.table_params import min_angle_isoceles_height,\
     table_min_internal_angle, minlegalA, minlegalB, maxlegalA, maxlegalB, min_intersect_angle

from tectosaur.kernels import kernels
%>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/pybind11_nparray.hpp"
#include "../include/vec_tensor.hpp"
#include "../include/timing.hpp"

namespace py = pybind11;

Vec3 get_split_pt(const Tensor3& tri) {
    auto base_vec = sub(tri[1], tri[0]);
    auto midpt = add(mult(base_vec, 0.5), tri[0]);
    auto to2 = sub(tri[2], midpt);
    auto V = sub(to2, projection(to2, base_vec));
    auto Vlen = length(V);
    auto base_len = length(base_vec);
    auto V_scaled = mult(V, base_len * ${min_angle_isoceles_height} / Vlen);
    return add(midpt, V_scaled);
}

Vec3 get_edge_lens(const Tensor3& tri) {
    Vec3 out;
    for (int d = 0; d < 3; d++) {
        out[d] = 0.0;
        for (int c = 0; c < 3; c++) {
            auto delta = tri[(d + 1) % 3][c] - tri[d][c];
            out[d] += delta * delta;
        }
    }
    return out;
}

int get_longest_edge(const Vec3& lens) {
    if (lens[0] >= lens[1] && lens[0] >= lens[2]) {
        return 0;
    } else if (lens[1] >= lens[0] && lens[1] >= lens[2]) {
        return 1;
    } else {// if (lens[2] >= lens[0] && lens[2] >= lens[1]) {
        return 2;
    }
}

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

double lawcos(double a, double b, double c) {
    return acos((a*a + b*b - c*c) / (2*a*b));
}

double rad2deg(double radians) {
    return radians * 180 / M_PI;
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

    // filter out when T1 < 20
    auto A1 = lawcos(1.0, L3, L2);
    if (rad2deg(A1) < angle_lim - eps) {
        return 3;
    }

    // filter out when A2 < 20
    auto A2 = lawcos(1.0, L2, L3);
    if (rad2deg(A2) < angle_lim - eps) {
        return 4;
    }

    // filter out when A3 < 20
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

struct StandardizeResult {
    int bad_tri_code;
    Tensor3 tri;
    std::array<int,3> labels;
    Vec3 translation;
    Tensor3 R;
    double scale;
};

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
    assert(0 == rot_out.first[2][2]);
    auto scale_out = scale(rot_out.first);
    int code = check_bad_tri(scale_out.first, angle_lim);
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

struct KernelProps {
    int scale_power;
    int sm_power;
    bool flip_negate;
};

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
        if (K == "${K.name}") { 
            return {
                ${K.scale_type}, ${K.sm_power}, ${str(K.flip_negate).lower()}
            };
        }
    % endfor
    else { throw std::runtime_error("unknown kernel"); }
}

std::array<double,81> transform_from_standard_pyshim(const std::array<double,81>& I,
    std::string K, double sm, const std::array<int,3>& labels,
    const Vec3& translation, const Tensor3& R, double scale) {

    return transform_from_standard(I, get_kernel_props(K), sm, labels, translation, R, scale);
}

NPArray<double> barycentric_evalnd_py(NPArray<double> pts, NPArray<double> wts, NPArray<double> vals, NPArray<double> xhat) {

    auto pts_buf = pts.request();
    auto vals_buf = vals.request();
    auto* pts_ptr = reinterpret_cast<double*>(pts.request().ptr);
    auto* wts_ptr = reinterpret_cast<double*>(wts.request().ptr);
    auto* vals_ptr = reinterpret_cast<double*>(vals.request().ptr);
    auto* xhat_ptr = reinterpret_cast<double*>(xhat.request().ptr);

    size_t n_out_dims = vals_buf.shape[1];
    size_t n_pts = pts_buf.shape[0];
    size_t n_dims = pts_buf.shape[1];

    auto out = make_array<double>({n_out_dims});
    auto* out_ptr = reinterpret_cast<double*>(out.request().ptr); 
    for (size_t out_d = 0; out_d < n_out_dims; out_d++) {
        double denom = 0;
        double numer = 0;
        for (size_t i = 0; i < n_pts; i++) {
            double kernel = 1.0;
            for (size_t d = 0; d < n_dims; d++) {
                double dist = xhat_ptr[d] - pts_ptr[i * n_dims + d];
                if (dist == 0) {
                    dist = 1e-16;
                }
                kernel *= dist;
            }
            kernel = wts_ptr[i] / kernel; 
            denom += kernel;
            numer += kernel * vals_ptr[i * n_out_dims + out_d];
        }
        out_ptr[out_d] = numer / denom;
    }
    return out;
}

double to_interval(double a, double b, double x) {
    return a + (b - a) * (x + 1.0) / 2.0;
}

double from_interval(double a, double b, double x) {
    return ((x - a) / (b - a)) * 2.0 - 1.0;
}

py::tuple coincident_lookup_pts(NPArray<double> tri_pts, double pr) {
    auto tri_pts_buf = tri_pts.request();
    auto* tri_pts_ptr = reinterpret_cast<double*>(tri_pts_buf.ptr);
    auto n_tris = tri_pts_buf.shape[0];

    auto out = make_array<double>({n_tris, 3});
    auto* out_ptr = reinterpret_cast<double*>(out.request().ptr);
    std::vector<StandardizeResult> standard_tris(n_tris);

#pragma omp parallel for
    for (size_t i = 0; i < n_tris; i++) {
        //TIC;
        Tensor3 tri;
        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                tri[d1][d2] = tri_pts_ptr[i * 9 + d1 * 3 + d2];
            }
        }
        //TOC("copy tri");

        //TIC2;
        auto standard_tri_info = standardize(tri, ${table_min_internal_angle}, true);
        standard_tris[i] = standard_tri_info;
        //TOC("standardize");

        //TIC2;
        double A = standard_tri_info.tri[2][0];
        double B = standard_tri_info.tri[2][1];

        double Ahat = from_interval(${minlegalA}, ${maxlegalA}, A);
        double Bhat = from_interval(${minlegalB}, ${maxlegalB}, B);
        double prhat = from_interval(0.0, 0.5, pr);

        out_ptr[i * 3] = Ahat;
        out_ptr[i * 3 + 1] = Bhat;
        out_ptr[i * 3 + 2] = prhat;
    }

    return py::make_tuple(out, standard_tris);
}

NPArray<double> coincident_lookup_from_standard(
    std::vector<StandardizeResult> standard_tris, 
    NPArray<double> interp_vals, NPArray<double> log_coeffs, std::string kernel, double sm)
{
    auto n_tris = standard_tris.size();
    auto out = make_array<double>({n_tris, 81});
    auto* out_ptr = reinterpret_cast<double*>(out.request().ptr);

    auto* interp_vals_ptr = reinterpret_cast<double*>(interp_vals.request().ptr);
    auto* log_coeffs_ptr = reinterpret_cast<double*>(log_coeffs.request().ptr);

    auto kernel_props = get_kernel_props(kernel);

#pragma omp parallel for
    for (size_t i = 0; i < n_tris; i++) {
        auto log_standard_scale = log(sqrt(length(tri_normal(standard_tris[i].tri))));
        
        std::array<double,81> interp_vals_array;
        for (int j = 0; j < 81; j++) {
            interp_vals_array[j] = 
                interp_vals_ptr[i * 81 + j] + log_standard_scale * log_coeffs_ptr[i * 81 + j];
        }

        auto transformed = transform_from_standard(
            interp_vals_array, kernel_props, sm,
            standard_tris[i].labels, standard_tris[i].translation,
            standard_tris[i].R, standard_tris[i].scale
        );

        for (int j = 0; j < 81; j++) {
            out_ptr[i * 81 + j] = transformed[j];
        }
    }
    return out;
}

Vec2 cramers_rule(const Vec2& a, const Vec2& b, const Vec2& c) {
    auto denom = a[0] * b[1] - b[0] * a[1];
    return {
        (c[0] * b[1] - b[0] * c[1]) / denom,
        (c[1] * a[0] - a[1] * c[0]) / denom
    };
}

// Solve least squares for a 3x2 system using cramers rule and the normal eqtns.
Vec2 least_squares_helper(const Vec3& a, const Vec3& b, const Vec3& c) {
    auto ada = dot(a, a);
    auto adb = dot(a, b);
    auto bdb = dot(b, b);

    auto adc = dot(a, c);
    auto bdc = dot(b, c);

    return cramers_rule({ada, adb}, {adb, bdb}, {adc, bdc});
}

Vec2 xyhat_from_pt(const Vec3& pt, const Tensor3& tri) {
    auto v1 = sub(tri[1], tri[0]);
    auto v2 = sub(tri[2], tri[0]);
    auto pt_trans = sub(pt, tri[0]);
        
    // Use cramer's rule to solve for xhat, yhat
    auto out = least_squares_helper(v1, v2, pt_trans);
    assert(out[0] + out[1] <= 1.0 + 1e-15);
    assert(out[0] >= -1e-15);
    assert(out[1] >= -1e-15);

    return out;
}

struct SeparateTriResult {
    std::array<Vec3,6> pts;
    std::array<std::array<int,3>,3> obs_tri;
    std::array<std::array<int,3>,3> src_tri;
    std::array<std::array<Vec2,3>,3> obs_basis_tri;
    std::array<std::array<Vec2,3>,3> src_basis_tri;
};

SeparateTriResult separate_tris(const Tensor3& obs_tri, const Tensor3& src_tri) {
    auto obs_split_pt = get_split_pt(obs_tri);
    auto obs_split_pt_xyhat = xyhat_from_pt(obs_split_pt, obs_tri);

    auto src_split_pt = get_split_pt(src_tri);
    auto src_split_pt_xyhat = xyhat_from_pt(src_split_pt, src_tri);

    std::array<Vec3,6> pts = {
        obs_tri[0], obs_tri[1], obs_tri[2], src_tri[2], obs_split_pt, src_split_pt
    };
    std::array<std::array<int,3>,3> obs_tris = {{{0,1,4}, {4,1,2}, {0,4,2}}};
    std::array<std::array<int,3>,3> src_tris = {{{1,0,5}, {5,0,3}, {1,5,3}}};
    std::array<std::array<Vec2,3>,3> obs_basis_tris = {{
        {{{0,0},{1,0},obs_split_pt_xyhat}},
        {{obs_split_pt_xyhat, {1,0},{0,1}}},
        {{{0,0},obs_split_pt_xyhat,{0,1}}}
    }};
    std::array<std::array<Vec2,3>,3> src_basis_tris = {{
        {{{0,0},{1,0},src_split_pt_xyhat}},
        {{src_split_pt_xyhat, {1,0},{0,1}}},
        {{{0,0},src_split_pt_xyhat,{0,1}}}
    }};

    return {pts, obs_tris, src_tris, obs_basis_tris, src_basis_tris};
}

py::tuple separate_tris_pyshim(const Tensor3& obs_tri, const Tensor3& src_tri) {
    auto out = separate_tris(obs_tri, src_tri);
    return py::make_tuple(
        out.pts, out.obs_tri, out.src_tri, out.obs_basis_tri, out.src_basis_tri
    );
}


Vec3 triangle_internal_angles(const Tensor3& tri) {
    auto v01 = sub(tri[1], tri[0]);
    auto v02 = sub(tri[2], tri[0]);
    auto v12 = sub(tri[2], tri[1]);

    auto L01 = length(v01);
    auto L02 = length(v02);
    auto L12 = length(v12);

    auto A1 = acos(dot(v01, v02) / (L01 * L02));
    auto A2 = acos(-dot(v01, v12) / (L01 * L12));
    auto A3 = M_PI - A1 - A2;

    return {A1, A2, A3};
}

double vec_angle(const Vec3& v1, const Vec3& v2) {
    auto v1L = length(v1);
    auto v2L = length(v2);
    auto v1d2 = dot(v1, v2);
    auto arg = v1d2 / (v1L * v2L);
    if (arg < -1) {
        arg = -1;
    } else if (arg > 1) {
        arg = 1;
    }
    return acos(arg);
}

double get_adjacent_phi(const Tensor3& obs_tri, const Tensor3& src_tri) {
    auto p = sub(obs_tri[1], obs_tri[0]);
    auto L1 = sub(obs_tri[2], obs_tri[0]);
    auto L2 = sub(src_tri[2], src_tri[0]);
    auto T1 = sub(L1, projection(L1, p));
    auto T2 = sub(L2, projection(L2, p));

    auto n1 = tri_normal(obs_tri);
    n1 = div(n1, length(n1));

    auto samedir = dot(n1, sub(T2, T1)) > 0;
    auto phi = vec_angle(T1, T2);

    if (samedir) {
        return phi;
    } else {
        return 2 * M_PI - phi;
    }
}

std::array<int,3> rotate_tri(int clicks) {
    return {clicks % 3, (clicks + 1) % 3, (clicks + 2) % 3};
}

std::array<std::array<int,3>,2> find_va_rotations(const std::array<int,3>& ot,
    const std::array<int,3>& st) 
{
    int ot_clicks = 0;
    int st_clicks = 0;
    bool matching_vert = false;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            matching_vert = st[d1] == ot[d2];
            if (matching_vert) {
                st_clicks = d1;
                ot_clicks = d2;
                break;
            }
        }
        if (matching_vert) {
            break;
        }
    }

    return {rotate_tri(ot_clicks), rotate_tri(st_clicks)};
}

struct VertexAdjacentSubTris {
    std::vector<Vec3> pts;
    std::vector<int> original_pair_idx;
    std::vector<std::array<int,3>> obs_tris;
    std::vector<std::array<int,3>> src_tris;
    std::vector<std::array<std::array<double,2>,3>> obs_basis;
    std::vector<std::array<std::array<double,2>,3>> src_basis;
};

struct EdgeAdjacentLookupTris {
    NPArray<double> pts;
    std::vector<std::array<std::array<double,2>,3>> obs_basis;
    std::vector<std::array<std::array<double,2>,3>> src_basis;
};

py::tuple adjacent_lookup_pts(NPArray<double> obs_tris, NPArray<double> src_tris,
    double pr, bool flip_symmetry) 
{
    VertexAdjacentSubTris va; 
    EdgeAdjacentLookupTris ea;


    auto obs_tris_buf = obs_tris.request();
    auto n_tris = obs_tris_buf.shape[0];

    ea.pts = make_array<double>({n_tris, 2});
    auto ea_pts_ptr = reinterpret_cast<double*>(ea.pts.request().ptr);

    auto* obs_tris_ptr = reinterpret_cast<double*>(obs_tris.request().ptr);
    auto* src_tris_ptr = reinterpret_cast<double*>(src_tris.request().ptr);
    
    for (int i = 0; i < static_cast<int>(n_tris); i++) {
        Tensor3 obs_tri;
        Tensor3 src_tri;
        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                obs_tri[d1][d2] = obs_tris_ptr[i * 9 + d1 * 3 + d2];
                src_tri[d1][d2] = src_tris_ptr[i * 9 + d1 * 3 + d2];
            }
        }

        auto sep_res = separate_tris(obs_tri, src_tri);

        ea.obs_basis.push_back(sep_res.obs_basis_tri[0]);
        ea.src_basis.push_back(sep_res.src_basis_tri[0]);

        for (size_t j = 0; j < sep_res.pts.size(); j++) {
            va.pts.push_back(sep_res.pts[j]);
        }
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                if (j == 0 && k == 0) {
                    continue;
                }

                auto otA = length(tri_normal({
                    sep_res.pts[sep_res.obs_tri[j][0]],
                    sep_res.pts[sep_res.obs_tri[j][1]],
                    sep_res.pts[sep_res.obs_tri[j][2]]
                }));

                auto stA = length(tri_normal({
                    sep_res.pts[sep_res.src_tri[k][0]],
                    sep_res.pts[sep_res.src_tri[k][1]],
                    sep_res.pts[sep_res.src_tri[k][2]]
                }));
                if (otA * stA < 1e-10) {
                    continue;
                }

                auto ot = sep_res.obs_tri[j];
                auto st = sep_res.src_tri[k];
                auto rot = find_va_rotations(ot, st);
                auto ot_rot = rot[0];
                auto st_rot = rot[1];

                va.original_pair_idx.push_back(i);
                va.obs_tris.push_back({
                    ot[ot_rot[0]] + 6 * i, ot[ot_rot[1]] + 6 * i, ot[ot_rot[2]] + 6 * i,
                });
                va.src_tris.push_back({
                    st[st_rot[0]] + 6 * i, st[st_rot[1]] + 6 * i, st[st_rot[2]] + 6 * i,
                });
                va.obs_basis.push_back({
                    sep_res.obs_basis_tri[j][ot_rot[0]],
                    sep_res.obs_basis_tri[j][ot_rot[1]],
                    sep_res.obs_basis_tri[j][ot_rot[2]],
                });
                va.src_basis.push_back({
                    sep_res.src_basis_tri[k][st_rot[0]],
                    sep_res.src_basis_tri[k][st_rot[1]],
                    sep_res.src_basis_tri[k][st_rot[2]],
                });
            }
        }

        auto phi = get_adjacent_phi(obs_tri, src_tri);
        double phi_max = M_PI;
        if (flip_symmetry) {
            if (phi > M_PI) {
                phi = 2 * M_PI - phi;
            }
            assert(${min_intersect_angle} <= phi && phi <= M_PI);
        } else {
            phi_max = 2 * M_PI - ${min_intersect_angle};
            assert(${min_intersect_angle} <= phi && phi <= 2 * M_PI - ${min_intersect_angle});
        }

        auto phihat = from_interval(${min_intersect_angle}, phi_max, phi);
        auto prhat = from_interval(0, 0.5, pr); 
        ea_pts_ptr[i * 2] = phihat;
        ea_pts_ptr[i * 2 + 1] = prhat;
    }
    return py::make_tuple(va, ea);
}

<%def name="basis_fnc(idx)">
% if idx == 0:
    1 - x - y
% elif idx == 1:
    x
% elif idx == 2:
    y
% endif
</%def>

std::array<double,81> sub_basis(const std::array<double,81>& I,
        const std::array<std::array<double,2>,3>& obs_basis_tri,
        const std::array<std::array<double,2>,3>& src_basis_tri)
{
    std::array<double,81> out{};
    % for ob1 in range(3):
    % for sb1 in range(3):
    % for ob2 in range(3):
    % for sb2 in range(3):

    {
        auto x = obs_basis_tri[${ob2}][0];
        auto y = obs_basis_tri[${ob2}][1];
        auto obv = ${basis_fnc(ob1)};

        x = src_basis_tri[${sb2}][0];
        y = src_basis_tri[${sb2}][1];
        auto sbv = ${basis_fnc(sb1)};

        (void)x;(void)y;

        % for d1 in range(3):
        % for d2 in range(3):

            <%
            out_idx = ob1 * 27 + d1 * 9 + sb1 * 3 + d2
            in_idx = ob2 * 27 + d1 * 9 + sb2 * 3 + d2
            %>
            out[${out_idx}] += I[${in_idx}] * obv * sbv;

        % endfor
        % endfor
    }

    % endfor
    % endfor
    % endfor
    % endfor
    return out;
}

void vert_adj_subbasis(NPArray<double> out, NPArray<double> Iv,
    const VertexAdjacentSubTris& va) 
{
    auto n_integrals = Iv.request().shape[0];    
    auto Iv_ptr = reinterpret_cast<double*>(Iv.request().ptr);
    auto out_ptr = reinterpret_cast<double*>(out.request().ptr);
#pragma omp parallel for
    for (size_t i = 0; i < n_integrals; i++) {
        std::array<double,81> this_integral;
        for (int j = 0; j < 81; j++) {
            this_integral[j] = Iv_ptr[i * 81 + j];
        }
        auto res = sub_basis(this_integral, va.obs_basis[i], va.src_basis[i]);

        int out_idx = va.original_pair_idx[i];
        for (int j = 0; j < 81; j++) {
#pragma omp atomic
            out_ptr[out_idx * 81 + j] += res[j];
        }
    }
}

NPArray<double> adjacent_lookup_from_standard(
    NPArray<double> obs_tris, NPArray<double> interp_vals, NPArray<double> log_coeffs,
    EdgeAdjacentLookupTris ea, std::string kernel, double sm)
{
    auto n_tris = obs_tris.request().shape[0];
    auto out = make_array<double>({n_tris, 81});
    auto* out_ptr = reinterpret_cast<double*>(out.request().ptr);

    auto* interp_vals_ptr = reinterpret_cast<double*>(interp_vals.request().ptr);
    auto* log_coeffs_ptr = reinterpret_cast<double*>(log_coeffs.request().ptr);
    auto* obs_tris_ptr = reinterpret_cast<double*>(obs_tris.request().ptr);

    auto kernel_props = get_kernel_props(kernel);

#pragma omp parallel for
    for (size_t i = 0; i < n_tris; i++) {
        Tensor3 tri;
        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                tri[d1][d2] = obs_tris_ptr[i * 9 + d1 * 3 + d2];
            }
        }
        auto standardized_res = standardize(tri, ${table_min_internal_angle}, false);

        auto log_standard_scale = log(sqrt(length(tri_normal(standardized_res.tri))));
        
        std::array<double,81> interp_vals_array;
        for (int j = 0; j < 81; j++) {
            interp_vals_array[j] = 
                interp_vals_ptr[i * 81 + j] + log_standard_scale * log_coeffs_ptr[i * 81 + j];
        }

        auto transformed = transform_from_standard(
            interp_vals_array, kernel_props, sm,
            standardized_res.labels, standardized_res.translation,
            standardized_res.R, standardized_res.scale
        );

        auto chunk = sub_basis(transformed, ea.obs_basis[i], ea.src_basis[i]);

        for (int j = 0; j < 81; j++) {
            out_ptr[i * 81 + j] = chunk[j];
        }
    }
    return out;
}

PYBIND11_PLUGIN(fast_lookup) {
    py::module m("fast_lookup");
    m.def("barycentric_evalnd", barycentric_evalnd_py); 
    
    m.def("get_edge_lens", get_edge_lens);
    m.def("get_longest_edge", get_longest_edge);
    m.def("get_origin_vertex", get_origin_vertex);
    m.def("relabel", relabel);
    m.def("check_bad_tri", check_bad_tri);
    m.def("rotation_matrix", rotation_matrix);
    m.def("translate", translate_pyshim);
    m.def("rotate1_to_xaxis", rotate1_to_xaxis_pyshim);
    m.def("rotate2_to_xyplane", rotate2_to_xyplane_pyshim);
    m.def("full_standardize_rotate", full_standardize_rotate_pyshim);
    m.def("scale", scale_pyshim);

    py::class_<StandardizeResult>(m, "StandardizeResult");
    m.def("standardize", standardize_pyshim);

    m.def("get_split_pt", get_split_pt);

    m.def("transform_from_standard", transform_from_standard_pyshim);
    m.def("sub_basis", sub_basis);

    
    m.def("coincident_lookup_pts", coincident_lookup_pts);
    m.def("coincident_lookup_from_standard", coincident_lookup_from_standard);
    m.def("adjacent_lookup_from_standard", adjacent_lookup_from_standard);

    m.def("xyhat_from_pt", xyhat_from_pt);
    m.def("separate_tris", separate_tris_pyshim);
    m.def("triangle_internal_angles", triangle_internal_angles);
    m.def("get_adjacent_phi", get_adjacent_phi);
    m.def("find_va_rotations", find_va_rotations);

    py::class_<VertexAdjacentSubTris>(m, "VertexAdjacentSubTris")
        .def_readonly("pts", &VertexAdjacentSubTris::pts)
        .def_readonly("original_pair_idx", &VertexAdjacentSubTris::original_pair_idx)
        .def_readonly("obs_tris", &VertexAdjacentSubTris::obs_tris)
        .def_readonly("src_tris", &VertexAdjacentSubTris::src_tris)
        .def_readonly("obs_basis", &VertexAdjacentSubTris::obs_basis)
        .def_readonly("src_basis", &VertexAdjacentSubTris::src_basis);

    py::class_<EdgeAdjacentLookupTris>(m, "EdgeAdjacentLookupTris")
        .def_readonly("pts", &EdgeAdjacentLookupTris::pts)
        .def_readonly("obs_basis", &EdgeAdjacentLookupTris::obs_basis)
        .def_readonly("src_basis", &EdgeAdjacentLookupTris::src_basis);

    m.def("adjacent_lookup_pts", adjacent_lookup_pts);
    m.def("vert_adj_subbasis", vert_adj_subbasis);

    return m.ptr();
}
