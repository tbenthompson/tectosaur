<% 
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-g', '-Wall', '-Werror'])
cfg['sources'] = ['fmm_impl.cpp', 'octree.cpp', 'blas_wrapper.cpp', 'cpp_tester.cpp']
cfg['dependencies'] = ['fmm_impl.hpp', 'octree.hpp', 'blas_wrapper.hpp']
cfg['parallel'] = True

import numpy as np
blas = np.__config__.blas_opt_info
cfg['library_dirs'] = blas['library_dirs']
cfg['libraries'] = blas['libraries']
%>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "fmm_impl.hpp"
#include "octree.hpp"

using namespace tectosaur;
namespace py = pybind11;

using NPArray = py::array_t<double,py::array::c_style>;

int main(int,char**);

std::vector<size_t> calc_strides(const std::vector<size_t>& shape, size_t unit_size) {
    std::vector<size_t> strides(shape.size());
    strides[shape.size() - 1] = unit_size;
    for (int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template <typename T>
pybind11::array_t<T> make_array(const std::vector<size_t>& shape) {
    return pybind11::array(pybind11::buffer_info(
        nullptr, sizeof(T), pybind11::format_descriptor<T>::value,
        shape.size(), shape, calc_strides(shape, sizeof(T))
    ));
}

template <typename T>
pybind11::array_t<T> array_from_vector(const std::vector<T>& in) {
    auto out = make_array<T>({in.size()});
    T* ptr = reinterpret_cast<T*>(out.request().ptr);
    for (size_t i = 0; i < in.size(); i++) {
        ptr[i] = in[i];
    }
    return out;
}


void check_shape(NPArray& arr) {
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("parameter requires n x 3 array.");
    }
}

std::vector<Vec3> get_vectors(NPArray& np_arr) {
    check_shape(np_arr);
    auto buf = np_arr.request();
    auto* first = reinterpret_cast<Vec3*>(buf.ptr);
    auto* last = first + buf.shape[0];
    return std::vector<Vec3>(first, last);
}


PYBIND11_PLUGIN(fmm) {
    py::module m("fmm");

    py::class_<Box>(m, "Box")
        .def_readonly("half_width", &Box::half_width)
        .def_readonly("center", &Box::center);

    py::class_<KDNode>(m, "KDNode")
        .def_readonly("start", &KDNode::start)
        .def_readonly("end", &KDNode::end)
        .def_readonly("bounds", &KDNode::bounds)
        .def_readonly("is_leaf", &KDNode::is_leaf)
        .def_readonly("idx", &KDNode::idx)
        .def_readonly("children", &KDNode::children);

    py::class_<KDTree>(m, "KDTree")
        .def("__init__", 
        [] (KDTree& kd, NPArray np_pts, NPArray np_normals, size_t n_per_cell) {
            check_shape(np_pts);
            check_shape(np_normals);
            new (&kd) KDTree(
                get_vectors(np_pts), get_vectors(np_normals),
                np_pts.request().shape[0], n_per_cell
            );
        })
        .def_readonly("nodes", &KDTree::nodes)
        .def_readonly("pts", &KDTree::pts);

    py::class_<SparseMat>(m, "SparseMat")
        .def("get_rows", [] (SparseMat& s) { return array_from_vector(s.rows); })
        .def("get_cols", [] (SparseMat& s) { return array_from_vector(s.cols); })
        .def("get_vals", [] (SparseMat& s) { return array_from_vector(s.vals); });

    py::class_<FMMMat>(m, "FMMMat")
        .def_readonly("p2p", &FMMMat::p2p)
        .def_readonly("p2m", &FMMMat::p2m)
        .def_readonly("m2p", &FMMMat::m2p)
        .def_readonly("m2m", &FMMMat::m2m)
        .def_readonly("multipole_starts", &FMMMat::multipole_starts);

    py::class_<FMMConfig>(m, "FMMConfig")
        .def("__init__", 
            [] (FMMConfig& cfg, double mac, NPArray surf, std::string k_name) {
                Kernel k{(k_name == "one") ? one : inv_r};
                new (&cfg) FMMConfig{mac, get_vectors(surf), std::move(k)};
            }
        );

    m.def("fmmmmmmm", &fmmmmmmm);

    m.def("run_tests", [] (std::vector<std::string> str_args) { 
        char** argv = new char*[str_args.size()];
        for (size_t i = 0; i < str_args.size(); i++) {
            argv[i] = const_cast<char*>(str_args[i].c_str());
        }
        main(str_args.size(), argv); 
        delete[] argv;
    });
    
    return m.ptr();
}
