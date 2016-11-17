<%
setup_pybind11(cfg)
%>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


///// start: pybind11_nparray.hpp
#include <pybind11/numpy.h>

template <typename T>
using NPArray = pybind11::array_t<T,pybind11::array::c_style>;
using NPArrayI = NPArray<int>;
using NPArrayF = NPArray<float>;
using NPArrayD = NPArray<double>;

std::vector<size_t> calc_strides(const std::vector<size_t>& shape, size_t unit_size) {
    std::vector<size_t> strides(shape.size());
    strides[shape.size() - 1] = unit_size;
    for (int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template <typename T>
NPArray<T> make_array(const std::vector<size_t>& shape) {
    return pybind11::array(pybind11::buffer_info(
        nullptr, sizeof(T), pybind11::format_descriptor<T>::value,
        shape.size(), shape, calc_strides(shape, sizeof(T))
    ));
}

template <typename T>
NPArray<T> array_from_vector(const std::vector<T>& in) {
    auto out = make_array<T>({in.size()});
    T* ptr = reinterpret_cast<T*>(out.request().ptr);
    for (size_t i = 0; i < in.size(); i++) {
        ptr[i] = in[i];
    }
    return out;
}

template <typename T, typename NPT>
T* as_ptr(NPArray<NPT>& np_arr) {
    return reinterpret_cast<T*>(np_arr.request().ptr);
}

template <typename T, typename NPT>
std::vector<T> get_vector(NPArray<NPT>& np_arr) {
    auto buf = np_arr.request();
    auto* first = reinterpret_cast<T*>(buf.ptr);
    auto* last = first + buf.shape[0];
    return std::vector<T>(first, last);
}
///// end: pybind11_nparray.hpp

namespace py = pybind11;

template <size_t D>
struct Cells {
    std::vector<std::array<double,D>> mins;
    std::vector<std::array<double,D>> maxs;
    std::vector<double> ests;
    int vector_dim;

    size_t size() const {
        return mins.size();
    }
};

template <size_t D>
Cells<D> initial_cell(std::array<double,D> min, std::array<double,D> max, NPArray<double> est) {
    int ests_size = est.request().size;
    std::vector<double> ests_vec(ests_size);
    auto* first_est = reinterpret_cast<double*>(est.request().ptr);
    for (int i = 0; i < ests_size; i++) {
        ests_vec[i] = first_est[i];
    }
    return Cells<D>{{min}, {max}, ests_vec, ests_size};
}

template <size_t D>
py::tuple get_subcell_mins_maxs(const Cells<D>& cells) {
    int splits = 1 << D;
    auto n_cells = cells.mins.size();
    auto cell_mins = make_array<double>({n_cells * splits, D});
    auto cell_maxs = make_array<double>({n_cells * splits, D});
    auto* first_min = reinterpret_cast<double*>(cell_mins.request().ptr);
    auto* first_max = reinterpret_cast<double*>(cell_maxs.request().ptr);
    for (size_t i = 0; i < n_cells; i++) {
        auto& min = cells.mins[i];
        auto& max = cells.maxs[i];
        std::array<double,D> center;
        std::array<double,D> width;
        for (size_t d = 0; d < D; d++) {
            center[d] = (min[d] + max[d]) / 2.0;
            width[d] = (max[d] - min[d]) / 2.0;
        }
        for (int ci = 0; ci < splits; ci++) {
            int mutable_ci = ci;

            std::array<int,D> bits;
            for (size_t d = 0; d < D; d++) {
                bits[d] = mutable_ci % 2;
                mutable_ci = mutable_ci >> 1;
            }
            // std::cout << bits[0] << " " << bits[1] << " " << bits[2] << std::endl;

            for (size_t d = 0; d < D; d++) {
                double non_center_val = center[d] + (bits[d] * 2 - 1) * width[d];
                auto np_idx = i * splits * D + ci * D + d;
                first_min[np_idx] = std::min(non_center_val, center[d]);
                first_max[np_idx] = std::max(non_center_val, center[d]);
            }
        }
    }

    return py::make_tuple(cell_mins, cell_maxs);
}

template <size_t D>
py::tuple refine(const Cells<D>& cells,
    NPArray<double> cell_mins, NPArray<double> cell_maxs,
    NPArray<double> cell_integrals, NPArray<double> iguess,
    int refine_step, int max_refinements) 
{
    int splits = 1 << D;

    auto* first_integral = reinterpret_cast<double*>(cell_integrals.request().ptr);
    auto* first_min = reinterpret_cast<double*>(cell_mins.request().ptr);
    auto* first_max = reinterpret_cast<double*>(cell_maxs.request().ptr);
    auto* iguess_ptr = reinterpret_cast<double*>(iguess.request().ptr);

    auto n_cells = cells.mins.size();

    Cells<D> out_cells;
    out_cells.vector_dim = cells.vector_dim;

    auto results = make_array<double>({size_t(cells.vector_dim)});
    auto* results_ptr = reinterpret_cast<double*>(results.request().ptr);
    for (int vec_dim = 0; vec_dim < cells.vector_dim; vec_dim++) {
        results_ptr[vec_dim] = 0;
    }

    std::vector<double> sums(cells.vector_dim);
    for (size_t i = 0; i < n_cells; i++) {
        auto idx_begin = i * splits;

        bool should_refine = false;
        for (int vec_dim = 0; vec_dim < cells.vector_dim; vec_dim++) {
            sums[vec_dim] = 0.0;
            for (int ci = 0; ci < splits; ci++) {
                sums[vec_dim] += first_integral[(idx_begin + ci) 
                    * cells.vector_dim + vec_dim];
            }
            double diff = cells.ests[i * cells.vector_dim + vec_dim] - sums[vec_dim];
            double iguess_val = iguess_ptr[vec_dim];
            if (iguess_val + diff != iguess_val) {
                should_refine = true;
            }
        }

        if (!should_refine || refine_step >= max_refinements - 1) {
            for (int vec_dim = 0; vec_dim < cells.vector_dim; vec_dim++) {
                results_ptr[vec_dim] += sums[vec_dim];
            }
            continue;
        }

        for (int ci = 0; ci < splits; ci++) {
            std::array<double,D> min;
            std::array<double,D> max;
            for (size_t dim = 0; dim < D; dim++) {
                min[dim] = first_min[(idx_begin + ci) * D + dim];
                max[dim] = first_max[(idx_begin + ci) * D + dim];
            }
            out_cells.mins.push_back(min);
            out_cells.maxs.push_back(max);

            for (int vec_dim = 0; vec_dim < cells.vector_dim; vec_dim++) {
                double val = first_integral[(idx_begin + ci) * cells.vector_dim + vec_dim];
                out_cells.ests.push_back(val);
            }
        }
    }

    return py::make_tuple(results, out_cells);
}

PYBIND11_PLUGIN(adapt_logic) {
    py::module m("adapt_logic", "");
    % for d in [1,2,3,4]:
    py::class_<Cells<${d}>>(m, "Cells${d}")
        .def("size", &Cells<${d}>::size);
    m.def("initial_cell${d}", initial_cell<${d}>);
    m.def("get_subcell_mins_maxs${d}", get_subcell_mins_maxs<${d}>);
    m.def("refine${d}", refine<${d}>);
    % endfor
    return m.ptr();
}
