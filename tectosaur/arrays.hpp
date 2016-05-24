#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace tectosaur {

using Real = float;
using Int = int;

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

template <typename T>
T* get_data(pybind11::array_t<T>& a) {
    return reinterpret_cast<T*>(a.request().ptr);
}

using Vec3 = std::array<Real,3>;
using Tri = std::array<std::array<Real,3>,3>;

inline Vec3 cross(const Vec3& x, const Vec3& y) {
    return {
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0]
    };
}

inline Vec3 sub(const Vec3& x, const Vec3& y) {
    return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

inline Vec3 get_unscaled_normal(const Tri& tri) {
    return cross(sub(tri[2], tri[0]), sub(tri[2], tri[1]));
}

inline Real magnitude(const Vec3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

} //end namespace tectosaur
