<%
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-g', '-Wall', '-Werror'])
%>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "lib/doctest.h"
#include "lib/test_helpers.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "taylor.hpp"
#include <iostream>

TEST_CASE("CreateConst") {
    auto t = Td<4>::constant(1.0);
    double exact[5] = {1.0, 0,0,0,0};
    REQUIRE_ARRAY_CLOSE(t.c, exact, 5, 1e-15);
}

TEST_CASE("CreateVar") {
    auto t = Td<4>::var(-2.0);
    double exact[5] = {-2.0, 1.0, 0,0,0};
    REQUIRE_ARRAY_CLOSE(t.c, exact, 5, 1e-15);
}

TEST_CASE("AddSub") {
    auto t = Td<4>::var(-4.0);
    auto res = 5 + t - 3 + t - 2 - t + t - t;
    double exact[5] = {-4.0, 1.0, 0,0,0};
    REQUIRE_ARRAY_CLOSE(res.c, exact, 5, 1e-15);
}

TEST_CASE("AddSub2") {
    auto t = Td<4>::var(-4.0);
    auto res = t + t;
    double exact[5] = {-8.0, 2.0, 0,0,0};
    REQUIRE_ARRAY_CLOSE(res.c, exact, 5, 1e-15);
}

TEST_CASE("AddSub3") {
    auto t = Td<4>::var(-4.0);
    auto res = 3 - t;
    double exact[5] = {7.0, -1.0, 0,0,0};
    REQUIRE_ARRAY_CLOSE(res.c, exact, 5, 1e-15);
}

TEST_CASE("Mul") {
    auto t = Td<7>::var(-1.5);
    auto t2 = t * t; 
    REQUIRE(t2.n_coeffs == 3);
    auto t4 = t2 * t2; 
    REQUIRE(t4.n_coeffs == 5);
    auto result = t4 * t2 * t + t2 * t + 8;
    double exact[8] = {
        -1595 / 128.0, 5535/64.0, -5247/32.0, 2851/16.0,
        -945/8.0, 189/4.0, -21/2.0, 1.0
    };
    REQUIRE(result.n_coeffs == 8);
    REQUIRE_ARRAY_CLOSE(result.c, exact, 8, 1e-15);
}

TEST_CASE("Div") {
    auto t = Td<7>::var(-1.5);
    auto t2 = t * t;
    auto divided = t2 / t;
    REQUIRE(divided.n_coeffs == 8);
    REQUIRE_ARRAY_CLOSE(t.c, divided.c, 8, 1e-15);
}

TEST_CASE("MulDivScalars") {
    auto t = Td<3>::var(3.0);
    auto res = 3 / t;
    double exact[4] = {1, -1 / 3.0, 1 / 9.0, -1.0 / 27.0};
    REQUIRE_ARRAY_CLOSE(res.c, exact, 4, 1e-15);
}

TEST_CASE("Sqrt") {
    auto t = Td<3>::var(3.0);
    auto res = sqrt(t);
    auto rt3 = std::sqrt(3);
    double exact[4] = {rt3, (1 / (2 * rt3)), -1 / (24 * rt3), 1 / (144 * rt3)};
    REQUIRE_ARRAY_CLOSE(res.c, exact, 4, 1e-15);
}

TEST_CASE("Log") {
    auto t = Td<3>::var(2);
    auto res = log(t);
    double exact[4] = {std::log(2), 1.0 / 2, -1.0 / 8, 1.0 / 24.0};
    REQUIRE_ARRAY_CLOSE(res.c, exact, 4, 1e-15);
}

TEST_CASE("Exp") {
    auto t = Td<3>::var(2);
    auto res = exp(log(t));
    double exact[4] = {2.0, 1.0, 0,0};
    REQUIRE_ARRAY_CLOSE(res.c, exact, 4, 1e-15);
    res = exp(t);
    auto exp2 = std::exp(2);
    double exact2[4] = {exp2, exp2, exp2 / 2, exp2 / 6};
    REQUIRE_ARRAY_CLOSE(res.c, exact2, 4, 1e-15);
}

TEST_CASE("Pow") {
    auto t = Td<3>::var(2);
    auto res = pow(t, 7/5.);
    auto A = std::pow(2, 2/5.);
    double exact[4] = {A * 2, A * (7 / 5.), A * (7 / 50.), -A * (7 / 500.)};
    REQUIRE_ARRAY_CLOSE(res.c, exact, 4, 1e-15);
}

TEST_CASE("Eval") {
    auto t = Td<3>::var(1);
    auto t2 = pow(t, 2);
    std::cout << t2 << std::endl;
    std::cout << t2.eval(3) << std::endl; 
}

template <typename T>
auto hyp(double obsx, T obsy, double srcx,
    std::array<double,2> n, std::array<double,2> N) 
{
    auto rx = obsx - srcx;
    auto ry = obsy;
    auto r2 = rx * rx + ry * ry;
    auto ndN = n[0] * N[0] + n[1] * N[1];
    auto ndr = n[0] * rx + n[1] * ry;
    auto Ndr = N[0] * rx + N[1] * ry;
    return (ndN - (2 * ndr * Ndr / r2)) / (2 * M_PI * r2);
}

TEST_CASE("Hypersingular") {
    std::vector<double> qx = {-0.99886640442,-0.994031969432,-0.985354084048,-0.972864385107,-0.956610955243,-0.936656618945,-0.913078556656,-0.885967979524,-0.85542976943,-0.821582070859,-0.7845558329,-0.744494302226,-0.701552468707,-0.655896465685,-0.607702927185,-0.557158304515,-0.504458144907,-0.449806334974,-0.393414311898,-0.335500245419,-0.27628819378,-0.216007236876,-0.154890589998,-0.0931747015601,-0.0310983383272,0.0310983383272,0.0931747015601,0.154890589998,0.216007236876,0.27628819378,0.335500245419,0.393414311898,0.449806334974,0.504458144907,0.557158304515,0.607702927185,0.655896465685,0.701552468707,0.744494302226,0.7845558329,0.821582070859,0.85542976943,0.885967979524,0.913078556656,0.936656618945,0.956610955243,0.972864385107,0.985354084048,0.994031969432,0.99886640442};
    std::vector<double> qw = {0.00290862255316,0.00675979919575,0.0105905483837,0.0143808227615,0.0181155607135,0.0217802431701,0.02536067357,0.0288429935805,0.0322137282236,0.0354598356151,0.0385687566126,0.0415284630901,0.0443275043388,0.0469550513039,0.0494009384495,0.0516557030696,0.053710621889,0.0555577448062,0.0571899256477,0.0586008498132,0.0597850587043,0.0607379708418,0.0614558995903,0.0619360674207,0.0621766166553,0.0621766166553,0.0619360674207,0.0614558995903,0.0607379708418,0.0597850587043,0.0586008498132,0.0571899256477,0.0555577448062,0.053710621889,0.0516557030696,0.0494009384495,0.0469550513039,0.0443275043388,0.0415284630901,0.0385687566126,0.0354598356151,0.0322137282236,0.0288429935805,0.02536067357,0.0217802431701,0.0181155607135,0.0143808227615,0.0105905483837,0.00675979919575,0.00290862255315};
    
    const int degree = 10;
    double dist = 0.4;
    auto obsy = Tf<degree>::var(dist);
    Tf<degree> result; 
    for (size_t i = 0; i < qx.size(); i++) {
        result += hyp(0.0, obsy, qx[i], {0, 1}, {0, 1}) * qw[i];
    }
    auto correct = -0.31830989;
    REQUIRE_CLOSE(result.eval(-dist), correct, 1e-5);
}

PYBIND11_PLUGIN(test_taylor) {
    pybind11::module m("test_taylor");

    m.def("run_tests", [] (std::vector<std::string> str_args) { 
        char** argv = new char*[str_args.size()];
        for (size_t i = 0; i < str_args.size(); i++) {
            argv[i] = const_cast<char*>(str_args[i].c_str());
        }
        main(str_args.size(), argv); 
        delete[] argv;
    });
    
    return m.ptr();
}
