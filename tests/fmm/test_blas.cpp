#include "blas_wrapper.hpp"
#include "doctest.h"
#include "test_helpers.hpp"

#include <iostream>

TEST_CASE("DGEMM test") {
    std::vector<double> A{0,1,2,3};
    std::vector<double> B{9,8,7,6};
    std::vector<double> correct{7, 6, 39, 34};
    auto result = mat_mult(2, 2, false, A, false, B);
    REQUIRE_ARRAY_EQUAL(result, correct, 4);
}

TEST_CASE("LU solve") 
{
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    auto lu = lu_decompose(matrix);
    auto soln = lu_solve(lu, {1,1});
    std::vector<double> correct{
        -0.25, 1.5
    };
    REQUIRE_ARRAY_CLOSE(soln, correct, 2, 1e-14);
}

TEST_CASE("SVD solve") 
{
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    auto svd = svd_decompose(matrix.data(), 2, 2);
    auto soln = svd_solve(svd, {1,1});
    std::vector<double> correct{
        -0.25, 1.5
    };
    REQUIRE_ARRAY_CLOSE(soln, correct, 2, 1e-14);
}

TEST_CASE("Condition number") 
{
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    auto svd = svd_decompose(matrix.data(), 2, 2);
    double cond = condition_number(svd);
    REQUIRE(cond == doctest::Approx(2.7630857945186595).epsilon(1e-12));
}

TEST_CASE("matrix vector product out pointer")
{
    
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    std::vector<double> vec{4, -2};
    std::vector<double> result{1, 1};
    matrix_vector_product(matrix.data(), 2, 2, vec.data(), result.data());
    REQUIRE_ARRAY_CLOSE(result, std::vector<double>{7, -4}, 2, 1e-15);
}

TEST_CASE("matrix vector product")
{
    
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    std::vector<double> vec{4, -2};
    auto result = matrix_vector_product(matrix.data(), 2, 2, vec.data());
    REQUIRE_ARRAY_CLOSE(result, std::vector<double>{6, -5}, 2, 1e-15);
}

TEST_CASE("matrix vector non-square")
{
    std::vector<double> matrix{
        2, 1, 1, -1, 0.5, 10
    };
    std::vector<double> vec{4,-2,0.5};
    auto result = matrix_vector_product(matrix.data(), 2, 3, vec.data());
    REQUIRE_ARRAY_CLOSE(result, std::vector<double>{6.5, 0}, 2, 1e-15);
}

TEST_CASE("matrix vector 0 columns")
{
    auto result = matrix_vector_product(nullptr, 0, 0, nullptr);
    REQUIRE(result.size() == size_t(0));
}

TEST_CASE("matvec") {
    BlockSparseMat m{{{1, 1, 2, 2, 0}}, {0, 2, 1, 3}};
    std::vector<double> in = {0, -1, 1};
    auto out = m.matvec(in.data(), 3);
    REQUIRE(out[0] == 0.0);
    REQUIRE(out[1] == 2.0);
    REQUIRE(out[2] == 2.0);
}
