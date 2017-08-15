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
struct ArrayMaker {
    static NPArray<T> make_array(const std::vector<size_t>& shape, T* buffer_ptr = nullptr) 
    {
        return pybind11::array(pybind11::buffer_info(
            buffer_ptr, sizeof(T), pybind11::format_descriptor<T>::value,
            shape.size(), shape, calc_strides(shape, sizeof(T))
        ));
    }
};

template <typename T, size_t dim>
struct ArrayMaker<std::array<T,dim>> {
    static NPArray<T> make_array(const std::vector<size_t>& shape_in, 
            std::array<T,dim>* buffer_ptr_in = nullptr) 
    {
        auto shape = shape_in;
        shape.push_back(dim);
        auto* buffer_ptr = reinterpret_cast<T*>(buffer_ptr_in);
        return pybind11::array(pybind11::buffer_info(
            buffer_ptr, sizeof(T), pybind11::format_descriptor<T>::value,
            shape.size(), shape, calc_strides(shape, sizeof(T))
        ));
    }
};

template <typename T>
auto make_array(const std::vector<size_t>& shape, T* buffer_ptr = nullptr) {
    return ArrayMaker<T>::make_array(shape, buffer_ptr);
}

template <typename T>
NPArray<T> array_from_vector(const std::vector<T>& in, std::vector<size_t> shape = {}) {
    if (shape.size() == 0) {
        shape = {in.size()};
    }
    auto out = make_array<T>(shape);
    assert(out.size() == in.size());
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

template <int D>
void check_shape(NPArrayD& arr) {
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != D) {
        std::string msg = "parameter requires n x ";
        msg += std::to_string(D);
        msg += " array.";
        throw std::runtime_error(msg);
    }
}

