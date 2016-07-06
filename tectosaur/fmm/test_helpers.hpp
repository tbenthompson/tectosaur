#pragma once
#include <cmath>

template <typename T1, typename T2>
void REQUIRE_ARRAY_EQUAL(const T1& a1, const T2& a2, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        REQUIRE(a1[i] == a2[i]);
    }
}

template <typename T1, typename T2>
void REQUIRE_ARRAY_CLOSE(const T1& a1, const T2& a2, size_t n, double epsilon)
{
    for (size_t i = 0; i < n; i++) {
        REQUIRE(std::fabs(a1[i] - a2[i]) < epsilon);
    }
}
