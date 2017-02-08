#include "kdtree.hpp"
#include <functional>

namespace tectosaur {

struct NBodyProblem {
    const Vec3* obs_pts;
    const Vec3* obs_ns;
    const Vec3* src_pts;
    const Vec3* src_ns;
    size_t n_obs;
    size_t n_src;
    const double* kernel_args;
};

struct Kernel {
    std::function<void(const NBodyProblem&,double*)> f;
    std::function<void(const NBodyProblem&,double*,double*)> f_mf;
    int tensor_dim;
};

Kernel get_by_name(std::string name);

} //end namespace tectosaur
