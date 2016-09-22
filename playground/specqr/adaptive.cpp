#include <iostream>
#include "adaptive_integrate.hpp"

/* Chebyshev nodes in the [a,b] interval. Good for interpolation. Avoids Runge
 * phenomenon. 
 */
std::vector<double> chebab(double a, double b, int n) {
    std::vector<double> out(n);
    for (int i = 0; i < n; i++) {
        out[i] = 0.5 * (a + b) + 0.5 * (b - a) * cos(((2 * i + 1) * M_PI) / (2 * n));
    }
    return out;
}

template <typename T>
T richardson_limit(double step_ratio, const std::vector<T>& values) 
{
    // assert(values.size() > 1);

    auto n_steps = values.size();
    auto last_level = values;
    decltype(last_level) this_level;

    for (size_t m = 1; m < n_steps; m++) {
        this_level.resize(n_steps - m);

        for (size_t i = 0; i < n_steps - m; i++) {
            auto mult = std::pow(step_ratio, m);
            auto factor = 1.0 / (mult - 1.0);
            auto low = last_level[i];
            auto high = last_level[i + 1];
            auto moreacc = factor * (mult * high - low);
            this_level[i] = moreacc;
        }
        last_level = this_level;
    }
    return this_level[0];
}

void testHinterp() {
    // Left right symmetry means that I only need to go from A in [0.0, 0.5]
    int nA = 16;
    int nB = 31;

    // These numbers are from ablims.py
    auto minlegalA = 0.100201758858 - 0.01;
    auto minlegalB = 0.24132345693 - 0.02;
    auto maxlegalB = 0.860758641203 + 0.02;
    auto Avals = chebab(minlegalA, 0.5, nA); 
    auto Bvals = chebab(minlegalB, maxlegalB, nB); 
    for (int i = 0; i < nA; i++) {
        double A = Avals[i];
        for (int j = 0; j < nB; j++) {
            double B = Bvals[j];
            Data d(1e-2, false, Hkernel, Tri{{{0,0,0},{1,0,0},{A,B,0}}}, 0.1, 1.0, 0.25);
            auto sum = integrate(d);
            std::cout << i << ", " << j << ", " 
                << A << ", " << B << ", " << sum[0] << ", " << std::endl;
        }
    }
}

void check(std::string k_name) {
    std::map<std::string,std::vector<double>> correct;
    correct["H"] = std::vector<double>{-0.0681271, 0.00374717, -2.92972e-08, -0.0479374, 0.000289359, -0.0310311, -0.0273078, 0.000289343, -0.0157399, 0.00374717, -0.0681271, 1.33654e-10, 0.000289359, -0.0273078, -0.0157399, 0.000289343, -0.0479374, -0.0310311, -2.92972e-08, 1.33654e-10, -0.253279, -0.0310311, -0.0157399, -0.0988838, -0.0157399, -0.0310311, -0.0988839, -0.0479374, 0.00028935, 0.0310311, -0.0773209, 0.0222243, 6.62088e-07, -0.0347615, 0.0145934, 0.0148439, 0.00028935, -0.0273078, 0.01574, 0.0222243, -0.0499281, -1.0635e-06, 0.0145934, -0.0347614, -0.014844, 0.0310311, 0.01574, -0.0988838, 6.62088e-07, -1.0635e-06, -0.25622, 0.0148439, -0.014844, -0.108641, -0.0273078, 0.000289349, 0.0157399, -0.0347616, 0.0145936, -0.0148446, -0.049928, 0.0222245, -1.30437e-06, 0.000289349, -0.0479374, 0.0310311, 0.0145936, -0.0347616, 0.0148445, 0.0222245, -0.0773207, 1.04316e-06, 0.0157399, 0.0310311, -0.0988839, -0.0148446, 0.0148445, -0.108641, -1.30437e-06, 1.04316e-06, -0.256222};
    correct["U"] = std::vector<double>{0.00627743, -0.00010931, 8.61569e-13, 0.00537344, -8.73445e-05, 0.000245357, 0.00502987, -8.73445e-05, 6.91567e-05, -0.00010931, 0.00627743, 6.53942e-13, -8.73445e-05, 0.00502987, 6.91567e-05, -8.73445e-05, 0.00537344, 0.000245357, 8.61569e-13, 6.53942e-13, 0.00602593, 0.000245357, 6.91567e-05, 0.00478726, 6.91567e-05, 0.000245357, 0.00478726, 0.00537344, -8.73445e-05, -0.000245357, 0.00617234, -0.000251347, -5.2341e-13, 0.00484596, -0.000324509, -0.000149677, -8.73445e-05, 0.00502987, -6.91567e-05, -0.000251347, 0.00596753, 2.48253e-13, -0.000324509, 0.00484596, 0.000149677, -0.000245357, -6.91567e-05, 0.00478726, -5.2341e-13, 2.48253e-13, 0.00581716, -0.000149677, 0.000149677, 0.00445606, 0.00502987, -8.73445e-05, -6.91567e-05, 0.00484596, -0.000324509, 0.000149677, 0.00596753, -0.000251347, 7.19921e-14, -8.73445e-05, 0.00537344, -0.000245357, -0.000324509, 0.00484596, -0.000149677, -0.000251347, 0.00617234, -2.4481e-13, -6.91567e-05, -0.000245357, 0.00478726, 0.000149677, -0.000149677, 0.00445606, 7.19921e-14, -2.4481e-13, 0.00581716};

    std::map<std::string,Kernel> Ks;
    Ks["H"] = Hkernel;
    Ks["U"] = Ukernel;

    Data d(1e-2, false, Ks[k_name], Tri{{{0,0,0},{1,0,0},{0,1,0}}}, 0.1, 1.0, 0.25);
    auto sum = integrate(d);

    for(int i = 0; i < 81; i++) {
        if (fabs(sum[i] - correct[k_name][i]) > 1e-4) {
            std::cout << "FAIL " << i << " " << sum[i] << " " << correct[k_name][i] << std::endl;
        }
    }
}

void limitU() {
    double eps0 = 0.01;
    double rich_step = 2;
    std::array<std::vector<double>,81> results;
    int n = 2;
    double eps = eps0;
    for (int i = 0; i < n; i++) {
        Data d(1e-5, false, Hkernel, Tri{{{0,0,0},{1,0,0},{0,1,0}}}, eps, 1.0, 0.25);
        auto I = integrate(d);
        for (int j = 0; j < 81; j++) {
            results[j].push_back(I[j]);
        }
        if (i > 0) {
            for (int j = 0; j < 81; j++) {
                double extrap = richardson_limit(rich_step, results[j]);
                std::cout << i << " " << j << " " << eps << " " << results[j].back() << " " << extrap << std::endl;
            }
        }
        eps /= rich_step;
    }
}

int main() {
    // check("U");
    // check("H");
    // limitU();

    Data d(1e-2, false, Ukernel, Tri{{{0,0,0},{1,0,0},{0.4,0.5,0}}}, 0.01, 1.0, 0.25);
    auto I = integrate(d);
    std::cout << I[0] << std::endl;
}

// correct result for H

// correct result for U
    
