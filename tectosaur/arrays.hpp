#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace tectosaur {

inline std::vector<size_t> calc_strides(
    const std::vector<size_t>& shape, size_t unit_size)
{
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
        nullptr,
        sizeof(T),
        pybind11::format_descriptor<T>::value(),
        shape.size(),
        shape,
        calc_strides(shape, sizeof(T))
    ));
}

} //end namespace tectosaur
