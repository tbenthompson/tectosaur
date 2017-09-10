#pragma once
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <array>

template <typename T1, typename T2>
void REQUIRE_ARRAY_EQUAL(const T1& a1, const T2& a2, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        REQUIRE(a1[i] == a2[i]);
    }
}

template <typename T1, typename T2>
void REQUIRE_CLOSE(const T1& a1, const T2& a2, double epsilon) {
    REQUIRE(std::fabs(a1 - a2) < epsilon);
}

template <typename T1, typename T2>
void REQUIRE_ARRAY_CLOSE(const T1& a1, const T2& a2, size_t n, double epsilon)
{
    for (size_t i = 0; i < n; i++) {
        REQUIRE_CLOSE(a1[i], a2[i], epsilon);
    }
}

template <size_t dim>
std::vector<std::array<double,dim>> random_pts(size_t N, double a = 0.0, double b = 1.0) 
{
    std::vector<std::array<double,dim>> out(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(a, b);
    for (size_t i = 0; i < N; i++) {
        for (size_t d = 0; d < dim; d++) {
            out[i][d] = dis(gen);
        }
    }
    return out;
}
