#pragma once
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

