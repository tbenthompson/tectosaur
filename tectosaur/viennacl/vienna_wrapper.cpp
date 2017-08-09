<%
setup_pybind11(cfg)
cfg['libraries'].extend(['OpenCL'])
cfg['dependencies'].extend(['../include/pybind11_nparray.hpp'])
%>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <CL/cl.h>

#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/device_specific/builtin_database/common.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp" 

#include "../include/pybind11_nparray.hpp"
#include "../include/timing.hpp"

namespace py = pybind11;
namespace vcl = viennacl;
namespace vclla = viennacl::linalg;

void check_platform() {
    typedef std::vector< vcl::ocl::platform > platforms_type;
    platforms_type platforms = vcl::ocl::get_platforms();
    bool is_first_element = true;
    for (platforms_type::iterator platform_iter  = platforms.begin();
        platform_iter != platforms.end();
        ++platform_iter)
    {
        typedef std::vector<vcl::ocl::device> devices_type;
        devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);
        std::cout << "# =========================================" << std::endl;
        std::cout << "#         Platform Information             " << std::endl;
        std::cout << "# =========================================" << std::endl;
        std::cout << "#" << std::endl;
        std::cout << "# Vendor and version: " << platform_iter->info() << std::endl;
        std::cout << "#" << std::endl;
        if (is_first_element)
        {
            std::cout << "# ViennaCL uses this OpenCL platform by default." << std::endl;
            is_first_element = false;
        }
        std::cout << "# " << std::endl;
        std::cout << "# Available Devices: " << std::endl;
        std::cout << "# " << std::endl;
        for (devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
        {
            std::cout << std::endl;
            std::cout << "  -----------------------------------------" << std::endl;
            std::cout << iter->full_info();
            std::cout << "ViennaCL Device Architecture:  " << iter->architecture_family() << std::endl;
            std::cout << "ViennaCL Database Mapped Name: " << vcl::device_specific::builtin_database::get_mapped_device_name(iter->name(), iter->vendor_id()) << std::endl;
            std::cout << "  -----------------------------------------" << std::endl;
        }
        std::cout << std::endl;
        std::cout << "###########################################" << std::endl;
        std::cout << std::endl;
    }
}

void setup(long ctx_ip, long device_ip, long queue_ip) {
    cl_context my_context = reinterpret_cast<cl_context>(ctx_ip);
    cl_device_id my_device = reinterpret_cast<cl_device_id>(device_ip);
    cl_command_queue my_queue = reinterpret_cast<cl_command_queue>(queue_ip);
    vcl::ocl::setup_context(0, my_context, my_device, my_queue);
}

PYBIND11_PLUGIN(vienna_wrapper) {
    py::module m("vienna_wrapper");
    m.def("check_platform", check_platform);
    m.def("setup", setup);
    m.def("mat_vec_prod_s", 
        [] (long A_ip, long x_ip, long y_ip, int m, int n) {
            cl_mem A_mem = reinterpret_cast<cl_mem>(A_ip);
            cl_mem x_mem = reinterpret_cast<cl_mem>(x_ip);
            cl_mem y_mem = reinterpret_cast<cl_mem>(y_ip);

            vcl::matrix<float> A(A_mem, m, n);
            vcl::vector<float> x(x_mem, n);
            vcl::vector<float> y(y_mem, m);
            vclla::prod_impl(A, x, y);
        }
    );

    m.def("mat_mat_prod_s", 
        [] (long A_ip, long B_ip, long C_ip, int m, int n, int r) {
            cl_mem A_mem = reinterpret_cast<cl_mem>(A_ip);
            cl_mem B_mem = reinterpret_cast<cl_mem>(B_ip);
            cl_mem C_mem = reinterpret_cast<cl_mem>(C_ip);

            vcl::matrix<float> A(A_mem, m, n);
            vcl::matrix<float> B(B_mem, n, r);
            vcl::matrix<float> C(C_mem, m, r);
            float alpha = 1.0;
            float beta = 0.0;
            vclla::prod_impl(A, B, C, alpha, beta);
        }
    );
        
    return m.ptr();
}
