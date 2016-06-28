<% 
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-g', '-Wall', '-Werror', '-DTASKLOAF_DEBUG'])
cfg['sources'] = ['fmm_impl.cpp', 'octree.cpp', 'blas_wrapper.cpp', 'cpp_tester.cpp']
cfg['dependencies'] = ['fmm_impl.hpp', 'octree.hpp', 'blas_wrapper.hpp']
cfg['parallel'] = True
cfg['include_dirs'].append('/home/tbent/projects/taskloaf/taskloaf/src/')
cfg['include_dirs'].append('/home/tbent/projects/taskloaf/taskloaf/lib/')
taskloaf_lib_dir = '/home/tbent/projects/taskloaf/taskloaf/taskloaf'
cfg['library_dirs'] = [taskloaf_lib_dir]
cfg['linker_args'] = ['-Wl,-rpath,' + taskloaf_lib_dir]
cfg['libraries'] = [':wrapper.cpython-35m-x86_64-linux-gnu.so']

import numpy as np
blas = np.__config__.blas_opt_info
cfg['library_dirs'].extend(blas['library_dirs'])
cfg['libraries'].extend(blas['libraries'])
%>
//TODO: taskloaf should provide some kind of configuration grabber like pybind11
//this would solve the bad configuration seg fault problem
//being header only could help

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "taskloaf.hpp"
#include "fmm_impl.hpp"
#include "octree.hpp"

using namespace tectosaur;
namespace py = pybind11;

int main(int,char**);

std::vector<Vec3> get_vectors(py::array_t<double>& np_arr) {
    auto buf = np_arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("parameter requires n x 3 array.");
    }
    auto* first = reinterpret_cast<Vec3*>(buf.ptr);
    auto* last = first + buf.shape[0];
    return std::vector<Vec3>(first, last);
};


PYBIND11_PLUGIN(fmm) {
    py::module m("fmm");

    py::class_<Box>(m, "Box")
        .def_readonly("half_width", &Box::half_width)
        .def_readonly("center", &Box::center);

    py::class_<NodeData>(m, "NodeData")
        .def_readonly("original_indices", &NodeData::original_indices)
        .def_readonly("pts", &NodeData::pts)
        .def_readonly("normals", &NodeData::normals);

    py::class_<OctreeNode>(m, "OctreeNode")
        .def_readonly("bounds", &OctreeNode::bounds)
        .def_readonly("data", &OctreeNode::data)
        .def_readonly("is_leaf", &OctreeNode::is_leaf)
        .def("get_child", [] (OctreeNode& n, int i) -> OctreeNode& {
            return *n.children[i].get(); 
        });

    py::class_<Octree>(m, "Octree")
        .def_property_readonly("root", [] (Octree& o) -> OctreeNode& {
            return *o.root.get(); 
        });

    m.def("make_octree", [] (size_t max_pts_per_cell, py::array_t<double> np_pts,
            py::array_t<double> np_normals) 
        {
            if (max_pts_per_cell < 1) {
                throw std::runtime_error("Need at least one point per cell.");
            }

            NodeData data;
            data.pts = get_vectors(np_pts);
            data.normals = get_vectors(np_normals);
            data.original_indices.resize(data.pts.size());
            std::iota(data.original_indices.begin(), data.original_indices.end(), 0);

            return Octree(max_pts_per_cell, std::move(data));
        }
    );

    m.def("run_tests", [] (std::vector<std::string> str_args) { 
        char** argv = new char*[str_args.size()];
        for (size_t i = 0; i < str_args.size(); i++) {
            argv[i] = const_cast<char*>(str_args[i].c_str());
        }
        main(str_args.size(), argv); 
        delete[] argv;
    });

    m.def("n_total_children", &n_total_children);

    py::class_<Upward>(m, "Upward");
    m.def("up_up_up",
        [] (Octree o, py::array_t<double> np_fmm_surf) {
            auto fmm_surf = get_vectors(np_fmm_surf); 
            return up_up_up(
                o, 
                std::make_shared<decltype(fmm_surf)>(std::move(fmm_surf))
            );
        }
    );

    m.def("go_go_go", &go_go_go);

    return m.ptr();
}
