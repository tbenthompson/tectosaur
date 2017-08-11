#pragma once
#include <array>
#include <cmath>

//TODO: This and vec_tensor.hpp can be merged some.
template <size_t dim>
inline double dot(const std::array<double,dim>& a, const std::array<double,dim>& b) {
    double out = 0;
    for (size_t d = 0; d < dim; d++) {
        out += a[d] * b[d];
    }
    return out;
}

#define BINOPVECVEC(name,op) \
    template <size_t dim>\
    inline std::array<double,dim> name(const std::array<double,dim>& a,\
            const std::array<double,dim>& b) {\
        std::array<double,dim> out;\
        for (size_t d = 0; d < dim; d++) {\
            out[d] = a[d] op b[d];\
        }\
        return out;\
    }

#define BINOPVECSCALAR(name,op) \
    template <size_t dim>\
    inline std::array<double,dim> name(const std::array<double,dim>& a, double b) {\
        std::array<double,dim> out;\
        for (size_t d = 0; d < dim; d++) {\
            out[d] = a[d] op b;\
        }\
        return out;\
    }

#define BINOP(name, op)\
    BINOPVECVEC(name, op)\
    BINOPVECSCALAR(name, op)

BINOP(add,+)
BINOP(sub,-)
BINOP(mult,*)
BINOP(div,/)

#undef BINOPVECVEC
#undef BINOPVECSCALAR
#undef BINOP

template <size_t dim>
inline double hypot2(const std::array<double,dim>& v) {
    return dot(v, v);
}

template <size_t dim>
inline double hypot(const std::array<double,dim>& v) {
    return std::sqrt(hypot2(v));
}

template <size_t dim>
inline double dist2(const std::array<double,dim>& a, const std::array<double,dim>& b) {
    return hypot2(sub(a,b));
}

template <size_t dim>
inline double dist(const std::array<double,dim>& a, const std::array<double,dim>& b) {
    return hypot(sub(a,b));
}

template <size_t dim>
struct Ball {
    std::array<double,dim> center;
    double R;

    Ball() = default;
    Ball(std::array<double,dim> center, double R): center(center), R(R) {}

};

template <size_t dim>
double R_to_box_width(double R) {
    return R / std::sqrt(static_cast<double>(dim));
}

template <size_t dim>
double box_width_to_R(double width) {
    return width * std::sqrt(static_cast<double>(dim));
}

template <size_t dim>
std::array<size_t,dim> make_child_idx(size_t i) 
{
    std::array<size_t,dim> child_idx;
    for (int d = dim - 1; d >= 0; d--) {
        auto idx = i % 2;
        i = i >> 1;
        child_idx[d] = idx;
    }
    return child_idx;
}

template <size_t dim>
Ball<dim> get_subbox(const Ball<dim>& b, const std::array<size_t,dim>& idx)
{
    // We get the sub-cell from an octree perspective. But, because we store balls
    // instead of boxes, we first need to convert into box widths to offset the center.
    auto width = R_to_box_width<dim>(b.R);
    auto new_width = width / 2.0;
    auto new_center = b.center;
    for (size_t d = 0; d < dim; d++) {
        new_center[d] += ((static_cast<double>(idx[d]) * 2) - 1) * new_width;
    }
    return {new_center, box_width_to_R<dim>(new_width)};
}

template <size_t dim>
int find_containing_subcell(const Ball<dim>& b, const std::array<double,dim>& pt) {
    int child_idx = 0;
    for (size_t d = 0; d < dim; d++) {
        if (pt[d] > b.center[d]) {
            child_idx++; 
        }
        if (d < dim - 1) {
            child_idx = child_idx << 1;
        }
    }
    return child_idx;
}

template <size_t dim>
bool in_ball(const Ball<dim>& b, const std::array<double,dim>& pt) {
    return dist(pt, b.center) <= b.R;
}

