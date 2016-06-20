<% setup_pybind11(cfg); cfg['compiler_args'].append('-std=c++14') %>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

std::vector<size_t> calc_strides(const std::vector<size_t>& shape, size_t unit_size)
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
        pybind11::format_descriptor<T>::value,
        shape.size(),
        shape,
        calc_strides(shape, sizeof(T))
    ));
}

template <typename T>
T* data(pybind11::array_t<T>& a) {
    return reinterpret_cast<T*>(a.request().ptr);
}

template <typename T>
auto shape(pybind11::array_t<T>& a) {
    return a.request().shape;
}

using quad_rule = std::pair<pybind11::array_t<double>,pybind11::array_t<double>>;

struct quad_pt {
    double ox;
    double oy;
    double sx;
    double sy;
};

struct subtri_desc {
    std::function<double(double,double)> theta_min;
    std::function<double(double,double)> theta_max;
    std::function<double(double,double,double)> rho_max;
};

struct quad_pt1d {
    double x;
    double w;
};

quad_pt1d map_to(quad_rule qr, size_t i, double xmin, double xmax) {
    return {
        xmin + (xmax - xmin) * ((data(qr.first)[i] + 1) / 2),
        (xmax - xmin) * (data(qr.first)[i] / 2)
    };
}

quad_rule coincident_quad(quad_rule outer_sing_q1, quad_rule outer_sing_q23,
    quad_rule outer_smooth_q, quad_rule theta_q, quad_rule rho_q) 
{
    auto theta_lims = std::make_tuple(
        [] (double x, double y) { return M_PI - std::atan((1 - y) / x); },
        [] (double x, double y) { return M_PI + std::atan(y / x); },
        [] (double x, double y) { return 2 * M_PI - std::atan(y / (1 - x)); }
    );

    auto rho_lims = std::make_tuple(
        [] (double x, double y, double t) {
            return (1 - y - x) / (std::cos(t) + std::sin(t));
        },
        [] (double x, double y, double t) {
            return -x / std::cos(t);
        },
        [] (double x, double y, double t) {
            return -y / std::sin(t);
        }
    );

    std::vector<quad_pt> pts;
    std::vector<double> wts;

    auto inner_integral = [&](double ox, double oy,
        double wxy, const subtri_desc& desc) 
    {
        auto theta_min = desc.theta_min(ox, oy);
        auto theta_max = desc.theta_max(ox, oy);
        for (size_t ti = 0; ti < shape(theta_q.first)[0]; ti++) {
            auto txw = map_to(theta_q, ti, theta_min, theta_max);
            auto rho_max = desc.rho_max(ox, oy, txw.x);
            for (size_t ri = 0; ri < shape(rho_q.first)[0]; ri++) {
                auto rxw = map_to(rho_q, ri, 0, rho_max);

                // Convert back to cartesian for source coords.
                auto sx = ox + rxw.x * std::cos(txw.x);
                auto sy = oy + rxw.x * std::sin(txw.x);
                auto jacobian = rxw.x;
                pts.push_back({ox, oy, sx, sy});
                wts.push_back(wxy * txw.x * rxw.x * jacobian);
            }
        }
    };

    auto outer_integral23 = [&] (quad_rule& ox_quad,
        quad_rule& oy_quad, const subtri_desc& desc) 
    {
        for (size_t xi = 0; xi < shape(ox_quad.first)[0]; xi++) {
            auto xw = map_to(ox_quad, xi, 0, 1);
            for (size_t yi = 0; yi < shape(oy_quad.first)[0]; yi++) {
                auto yw = map_to(oy_quad, yi, 0, 1 - xw.x);
                inner_integral(xw.x, yw.x, xw.w * yw.w, desc);         
            }
        }
    };

    outer_integral23(outer_sing_q23, outer_smooth_q, 
        subtri_desc{
            std::get<0>(theta_lims),
            std::get<1>(theta_lims),
            std::get<1>(rho_lims)
        }
    );

    outer_integral23(outer_sing_q23, outer_smooth_q, 
        subtri_desc{
            std::get<1>(theta_lims),
            std::get<2>(theta_lims),
            std::get<2>(rho_lims)
        }
    );
    
    return outer_sing_q1;
    // auto result = make_array<double>({10}); 
    // for (int i = 0; i < 10; i++) {
    //     data(result)[i] = i;
    // }
    // return result;
}

PYBIND11_PLUGIN(_triangle_rules) {
    pybind11::module m("_triangle_rules", "");
    m.def("coincident_quad", &coincident_quad);
    return m.ptr();
}
