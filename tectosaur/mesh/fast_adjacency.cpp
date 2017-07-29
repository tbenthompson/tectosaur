<%
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-O3', '-g', '-UNDEBUG', '-DDEBUG'])
cfg['dependencies'].append('../include/pybind11_nparray.hpp')
%>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/pybind11_nparray.hpp"

namespace py = pybind11;

std::vector<std::vector<std::pair<size_t,int>>> find_touching_pts(long* tris, size_t n_tris) {
    size_t n_pts = *std::max_element(tris, tris + (n_tris * 3)) + 1;
    std::vector<std::vector<std::pair<size_t,int>>> out(n_pts);
    for (size_t i = 0; i < n_tris; i++) {
        for (int d = 0; d < 3; d++) {
            out[tris[i * 3 + d]].push_back({i, d});
        }
    }
    return out;
}

std::pair<std::vector<size_t>,std::vector<size_t>> 
find_adjacents(long* tris, size_t n_tris) {
    auto touching_pts = find_touching_pts(tris, n_tris); 
    std::vector<size_t> va;
    std::vector<size_t> ea;
    for (size_t i = 0; i < n_tris; i++) {
        std::vector<std::tuple<size_t,int,size_t>> touching_tris;
        for (int d = 0; d < 3; d++) {
            for (auto& other_t: touching_pts[tris[i * 3 + d]]) {
                touching_tris.push_back({other_t.first,d,other_t.second});
            }
        }

        std::vector<size_t> already;
        for (auto& other_t: touching_tris) {
            auto other_t_idx = std::get<0>(other_t);
            if (other_t_idx == i ||
                std::find(already.begin(), already.end(), other_t_idx) != already.end())
            {
                continue;
            }
            already.push_back(other_t_idx); 

            std::pair<size_t,size_t> stored_shared_vert;
            bool made_va = false;
            bool made_ea = false;
            for (auto& other_t2: touching_tris) {
                if (std::get<0>(other_t2) != other_t_idx) {
                    continue;
                }
                if (made_ea) {
                    throw std::runtime_error("Duplicate triangles!");
                }
                auto shared_vert = std::pair<size_t,size_t>{
                    std::get<1>(other_t2), std::get<2>(other_t2)
                };
                if (!made_va) {
                    stored_shared_vert = shared_vert;
                    made_va = true;
                } else {
                    ea.push_back(i);
                    ea.push_back(other_t_idx);
                    ea.push_back(stored_shared_vert.first);
                    ea.push_back(stored_shared_vert.second);
                    ea.push_back(shared_vert.first);
                    ea.push_back(shared_vert.second);
                    made_ea = true;
                }
            }
            if (!made_ea) {
                assert(made_va);
                va.push_back(i);
                va.push_back(other_t_idx);
                va.push_back(stored_shared_vert.first);
                va.push_back(stored_shared_vert.second);
            }
        }
    }
    return std::make_pair(std::move(va), std::move(ea));
}

// std::pair<std::vector<size_t>,std::vector<size_t>> 
// find_adjacents2(long* tris, size_t n_tris) {
//     for (size_t i = 0; i < n_tris; i++) {
//          
//     }
// }

PYBIND11_PLUGIN(fast_adjacency) {
    py::module m("fast_adjacency", "");

    m.def("find_touching_pts", 
        [] (NPArray<long> tris) {
            auto tris_ptr = as_ptr<long>(tris);
            return find_touching_pts(tris_ptr, tris.request().shape[0]);
        });
    m.def("find_adjacents", 
        [] (NPArray<long> tris) {
            auto tris_ptr = as_ptr<long>(tris);
            auto out = find_adjacents(tris_ptr, tris.request().shape[0]);
            return py::make_tuple(
                array_from_vector(out.first, {out.first.size() / 4, 4}),
                array_from_vector(out.second, {out.second.size() / 6, 6})
            );
        });


    return m.ptr();
}
