<% 
from tectosaur.util.build_cfg import fmm_lib_cfg
fmm_lib_cfg(cfg)
%>

#include "include/pybind11_nparray.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>


#include "fmm_impl.hpp"
#include "octree.hpp"

namespace py = pybind11;

template <size_t dim>
void wrap_dim(py::module& m) {
    m.def("surrounding_surface", surrounding_surface<dim>);
    m.def("inscribe_surf", &inscribe_surf<dim>);
    m.def("c2e_solve", &c2e_solve<dim>);

    m.def("in_box", &in_box<dim>);

    py::class_<Cube<dim>>(m, "Cube")
        .def(py::init<std::array<double,dim>, double>())
        .def_readonly("center", &Cube<dim>::center)
        .def_readonly("width", &Cube<dim>::width);


    py::class_<OctreeNode<dim>>(m, "OctreeNode")
        .def_readonly("start", &OctreeNode<dim>::start)
        .def_readonly("end", &OctreeNode<dim>::end)
        .def_readonly("bounds", &OctreeNode<dim>::bounds)
        .def_readonly("is_leaf", &OctreeNode<dim>::is_leaf)
        .def_readonly("idx", &OctreeNode<dim>::idx)
        .def_readonly("height", &OctreeNode<dim>::height)
        .def_readonly("depth", &OctreeNode<dim>::depth)
        .def_readonly("children", &OctreeNode<dim>::children);

    py::class_<Octree<dim>>(m, "Octree")
        .def("__init__",
        [] (Octree<dim>& kd, NPArrayD np_pts, size_t n_per_cell) {
            check_shape<dim>(np_pts);
            new (&kd) Octree<dim>(
                reinterpret_cast<std::array<double,dim>*>(np_pts.request().ptr),
                np_pts.request().shape[0], n_per_cell
            );
        })
        .def("root", &Octree<dim>::root)
        .def_readonly("nodes", &Octree<dim>::nodes)
        .def_readonly("orig_idxs", &Octree<dim>::orig_idxs)
        .def_readonly("max_height", &Octree<dim>::max_height)
        .def_property_readonly("pts", [] (Octree<dim>& tree) {
            return make_array<double>(
                {tree.pts.size(), dim},
                reinterpret_cast<double*>(tree.pts.data())
            );
        })
        .def_property_readonly("n_nodes", [] (Octree<dim>& o) {
            return o.nodes.size();
        });

    py::class_<FMMConfig<dim>>(m, "FMMConfig")
        .def("__init__", 
            [] (FMMConfig<dim>& cfg, double equiv_r,
                double check_r, size_t order, std::string k_name,
                NPArrayD params) 
            {
                new (&cfg) FMMConfig<dim>{
                    equiv_r, check_r, order, get_by_name<dim>(k_name),
                    get_vector<double>(params)                                    
                };
            }
        )
        .def_readonly("inner_r", &FMMConfig<dim>::inner_r)
        .def_readonly("outer_r", &FMMConfig<dim>::outer_r)
        .def_readonly("order", &FMMConfig<dim>::order)
        .def_readonly("params", &FMMConfig<dim>::params)
        .def_property_readonly("kernel_name", &FMMConfig<dim>::kernel_name)
        .def_property_readonly("tensor_dim", &FMMConfig<dim>::tensor_dim);

#define EVALFNC(FNCNAME)\
        def(#FNCNAME"_eval", [] (FMMMat<dim>& m, NPArrayD out, NPArrayD in) {\
            auto* out_ptr = reinterpret_cast<double*>(out.request().ptr);\
            auto* in_ptr = reinterpret_cast<double*>(in.request().ptr);\
            m.FNCNAME##_matvec(out_ptr, in_ptr);\
        })
#define EVALFNCLEVEL(FNCNAME)\
        def(#FNCNAME"_eval", [] (FMMMat<dim>& m, NPArrayD out, NPArrayD in, int level) {\
            auto* out_ptr = reinterpret_cast<double*>(out.request().ptr);\
            auto* in_ptr = reinterpret_cast<double*>(in.request().ptr);\
            m.FNCNAME##_matvec(out_ptr, in_ptr, level);\
        })
#define OP(NAME)\
        def_readonly(#NAME, &FMMMat<dim>::NAME)

    py::class_<FMMMat<dim>>(m, "FMMMat")
        .def_readonly("obs_tree", &FMMMat<dim>::obs_tree)
        .def_readonly("src_tree", &FMMMat<dim>::src_tree)
        .def_property_readonly("obs_normals", [] (FMMMat<dim>& mat) {
            return make_array<double>(
                {mat.obs_normals.size(), dim},
                reinterpret_cast<double*>(mat.obs_normals.data())
            );
        })
        .def_property_readonly("src_normals", [] (FMMMat<dim>& mat) {
            return make_array<double>(
                {mat.src_normals.size(), dim},
                reinterpret_cast<double*>(mat.src_normals.data())
            );
        })
        .def_readonly("surf", &FMMMat<dim>::surf)
        .def_readonly("cfg", &FMMMat<dim>::cfg)
        .def_property_readonly("u2e_ops", [] (FMMMat<dim>& fmm) {
            return make_array<double>(
                {fmm.u2e_ops.size()}, reinterpret_cast<double*>(fmm.u2e_ops.data())
            );
        })
        .def_property_readonly("d2e_ops", [] (FMMMat<dim>& fmm) {
            return make_array<double>(
                {fmm.d2e_ops.size()}, reinterpret_cast<double*>(fmm.d2e_ops.data())
            );
        })
        .def_property_readonly("tensor_dim", &FMMMat<dim>::tensor_dim)
        .OP(p2m).OP(m2m).OP(p2l).OP(m2l).OP(l2l).OP(p2p).OP(m2p).OP(l2p).OP(u2e).OP(d2e)
        .EVALFNC(p2p).EVALFNC(p2m).EVALFNC(p2l).EVALFNC(m2l).EVALFNC(m2p).EVALFNC(l2p)
        .EVALFNCLEVEL(m2m).EVALFNCLEVEL(u2e).EVALFNCLEVEL(l2l).EVALFNCLEVEL(d2e);

#undef EXPOSEOP
#undef EVALFNC
#undef EVALFNCLEVEL

    m.def("fmmmmmmm", 
        [] (const Octree<dim>& obs_tree, NPArrayD obs_normals,
            const Octree<dim>& src_tree, NPArrayD src_normals,
            const FMMConfig<dim>& cfg) 
        {
            return fmmmmmmm(
                obs_tree, get_vector<std::array<double,dim>>(obs_normals),
                src_tree, get_vector<std::array<double,dim>>(src_normals),
                cfg
            );
        });

    <%
    direct_eval_data = [
        ("",["","n_obs_dofs * n_src_dofs",""]),
        ("mf_",[",NPArrayD input","n_obs_dofs",", as_ptr<double>(input)"])
    ]
    %>
    % for name, extra in direct_eval_data:
        m.def("${name}direct_eval", [](std::string k_name, NPArrayD obs_pts, NPArrayD obs_ns,
                                NPArrayD src_pts, NPArrayD src_ns, NPArrayD params${extra[0]}) {
            check_shape<dim>(obs_pts);
            check_shape<dim>(obs_ns);
            check_shape<dim>(src_pts);
            check_shape<dim>(src_ns);
            auto K = get_by_name<dim>(k_name);
            int n_obs_dofs = K.tensor_dim * obs_pts.request().shape[0];
            int n_src_dofs = K.tensor_dim * src_pts.request().shape[0];
            (void)n_src_dofs;
            std::vector<double> out(${extra[1]});
            K.${name}f({as_ptr<std::array<double,dim>>(obs_pts), as_ptr<std::array<double,dim>>(obs_ns),
               as_ptr<std::array<double,dim>>(src_pts), as_ptr<std::array<double,dim>>(src_ns),
               obs_pts.request().shape[0], src_pts.request().shape[0],
               as_ptr<double>(params)},
              out.data()${extra[2]});
            return array_from_vector(out);
        });
    % endfor
}

PYBIND11_PLUGIN(fmm) {
    py::module m("fmm");
    auto two = m.def_submodule("two");
    auto three = m.def_submodule("three");

    wrap_dim<2>(two);
    wrap_dim<3>(three);

    py::class_<BlockSparseMat>(m, "BlockSparseMat")
        .def("get_nnz", &BlockSparseMat::get_nnz)
        .def("matvec", [] (BlockSparseMat& s, NPArrayD v, size_t n_rows) {
            auto out = s.matvec(reinterpret_cast<double*>(v.request().ptr), n_rows);
            return array_from_vector(out);
        });


#define NPARRAYPROP(name)\
    def_property_readonly(#name, [] (MatrixFreeOp& op) {\
        return make_array({op.name.size()}, op.name.data());\
    })

    py::class_<MatrixFreeOp>(m, "MatrixFreeOp")
        .NPARRAYPROP(obs_n_start).NPARRAYPROP(obs_n_end).NPARRAYPROP(obs_n_idx)
        .NPARRAYPROP(src_n_start).NPARRAYPROP(src_n_end).NPARRAYPROP(src_n_idx);
#undef NPARRAYPROP

    return m.ptr();
}
