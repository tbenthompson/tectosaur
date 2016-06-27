<% 
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-g', '-Wall', '-Werror'])
cfg['sources'] = ['fmm_impl.cpp', 'octree.cpp', 'cpp_tester.cpp']
cfg['dependencies'] = ['fmm_impl.hpp', 'octree.hpp']
cfg['parallel'] = True
cfg['include_dirs'].append('/home/tbent/projects/taskloaf/taskloaf/src/')
cfg['include_dirs'].append('/home/tbent/projects/taskloaf/taskloaf/lib/')
taskloaf_lib_dir = '/home/tbent/projects/taskloaf/taskloaf/taskloaf'
cfg['library_dirs'] = [taskloaf_lib_dir]
cfg['linker_args'] = ['-Wl,-rpath,' + taskloaf_lib_dir]
cfg['libraries'] = [':wrapper.cpython-35m-x86_64-linux-gnu.so']
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


PYBIND11_PLUGIN(fmm) {
    py::module m("fmm");

    py::class_<Box>(m, "Box")
        .def_readonly("half_width", &Box::half_width)
        .def_readonly("center", &Box::center);

    py::class_<OctreeNode>(m, "OctreeNode")
        .def_readonly("bounds", &OctreeNode::bounds)
        .def_buffer([] (OctreeNode &n) {
            return py::buffer_info(
                n.pts.data(), sizeof(double), py::format_descriptor<double>::value,
                2, {n.pts.size(), 3}, {sizeof(double) * 3, sizeof(double)}
            );
        })
        .def_readonly("is_leaf", &OctreeNode::is_leaf)
        .def("get_child", [] (OctreeNode& n, int i) -> OctreeNode& {
            return *n.children[i].get(); 
        });

    py::class_<Octree>(m, "Octree")
        .def("__init__", 
            [] (Octree& oct, size_t max_pts_per_cell, py::array_t<double> np_pts) {
                auto buf = np_pts.request();
                if (max_pts_per_cell < 1) {
                    throw std::runtime_error("Need at least one point per cell.");
                }
                if (buf.ndim != 2 || buf.shape[1] != 3) {
                    throw std::runtime_error("Octree requires n x 3 array of points.");
                }
                auto* first = reinterpret_cast<Vec3*>(buf.ptr);
                auto* last = first + buf.shape[0];
                std::vector<Vec3> pts(first, last);
                new (&oct) Octree(max_pts_per_cell, pts);
            })
        .def_property_readonly("root", [] (Octree& o) -> OctreeNode& {
            return *o.root.get(); 
        });

    m.def("run_tests", [] (std::vector<std::string> str_args) { 
        char** argv = new char*[str_args.size()];
        for (size_t i = 0; i < str_args.size(); i++) {
            argv[i] = const_cast<char*>(str_args[i].c_str());
        }
        main(str_args.size(), argv); 
        delete[] argv;
    });

    m.def("n_total_children", &n_total_children);

    py::class_<upward_traversal>(m, "UpwardTraversal");
    m.def("up_up_up", &up_up_up);

    return m.ptr();
}
