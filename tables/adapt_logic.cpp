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

    size_t size() const {
        return mins.size();
    }
};

template <size_t D>
Cells<D> initial_cell(std::array<double,D> min, std::array<double,D> max, double est) {
    return Cells<D>{{min}, {max}, {est}};
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
    NPArray<double> cell_integrals, double iguess) 
{
    int splits = 1 << D;

    auto* first_integral = reinterpret_cast<double*>(cell_integrals.request().ptr);
    auto* first_min = reinterpret_cast<double*>(cell_mins.request().ptr);
    auto* first_max = reinterpret_cast<double*>(cell_maxs.request().ptr);

    auto n_cells = cells.mins.size();

    Cells<D> out_cells;
    double result = 0;
    for (size_t i = 0; i < n_cells; i++) {
        auto idx_begin = i * splits;

        double sum = 0.0;
        for (int ci = 0; ci < splits; ci++) {
            sum += first_integral[idx_begin + ci];
        }

        double diff = cells.ests[i] - sum;
        if (iguess + diff == iguess) {
            result += sum;
            continue;
        }

        for (int ci = 0; ci < splits; ci++) {
            out_cells.mins.push_back({
                first_min[(idx_begin + ci) * 3 + 0],
                first_min[(idx_begin + ci) * 3 + 1],
                first_min[(idx_begin + ci) * 3 + 2]
            });

            out_cells.maxs.push_back({
                first_max[(idx_begin + ci) * 3 + 0],
                first_max[(idx_begin + ci) * 3 + 1],
                first_max[(idx_begin + ci) * 3 + 2]
            });

            out_cells.ests.push_back(first_integral[idx_begin + ci]);
        }
    }

    return py::make_tuple(result, out_cells);
}

PYBIND11_PLUGIN(adapt_logic) {
    py::module m("adapt_logic", "");
    py::class_<Cells<3>>(m, "Cells3")
        .def("size", &Cells<3>::size);
    m.def("initial_cell", initial_cell<3>);
    m.def("get_subcell_mins_maxs", get_subcell_mins_maxs<3>);
    m.def("refine", refine<3>);
    return m.ptr();
}
