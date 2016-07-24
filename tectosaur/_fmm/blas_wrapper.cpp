#include "blas_wrapper.hpp"
#include "lib/doctest.h"
#include "test_helpers.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda, int* ipiv,
    int* info);
extern "C" void dgetrs_(char* TRANS, int* N, int* NRHS, double* A, int* LDA,
    int* IPIV, double* B, int* LDB, int* INFO);
extern "C" void dgesvd_(char* JOBU, char* JOBVT, int* M, int* N, double* A,
    int* LDA, double* S, double* U, int* LDU, double* VT, int* LDVT, double* WORK,
    int* LWORK, int* INFO);
extern "C" void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, 
    double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA,
    double* C, int* LDC);
extern "C" void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A,
    int* LDA, double* X, int* INCX, double* BETA, double* Y, int* INCY);

namespace tectosaur {

struct LU {
    std::vector<double> LU;
    std::vector<int> pivots;
};

void LUDeleter::operator()(LU* thing) {
    delete thing;
}


LUPtr lu_decompose(const std::vector<double>& matrix) 
{
    int n = std::sqrt(matrix.size()); 
    std::vector<double> A = matrix;
    std::vector<int> pivots(n);
    int info;
    dgetrf_(&n, &n, A.data(), &n, pivots.data(), &info);
    assert(info == 0);
    return LUPtr(new LU{A, pivots});
}

std::vector<double> lu_solve(const LUPtr& lu, const std::vector<double>& b) 
{
    std::vector<double> x = b;
    int n = b.size(); 
    int n_rhs = 1;
    int info;
    // type = 'T' means that we solve A^T x = b rather than Ax = b. This is good
    // because blas operates on column major data while this code sticks to 
    // row major data.
    char type = 'T';
    dgetrs_(&type, &n, &n_rhs, lu->LU.data(), &n, lu->pivots.data(), x.data(), &n, &info);
    assert(info == 0);
    return x;
}

struct SVD {
    std::vector<double> singular_values;
    std::vector<double> left_singular_vectors;
    std::vector<double> right_singular_vectors;
    double threshold;
};

void SVDDeleter::operator()(SVD* thing) {
    delete thing;
}

SVDPtr svd_decompose(double* matrix, int m, int n) {
    char jobu = 'A';
    char jobvt = 'A';
    auto k = std::min(m, n);
    auto svd = SVDPtr(new SVD{
        std::vector<double>(k),
        std::vector<double>(m * m),
        std::vector<double>(n * n),
        1e-16
    });
    int lwork = std::max(
        std::max(1, 3 * std::min(m, n) + std::max(m, n)),
        5 * std::min(m, n)
    );
    std::vector<double> work_space(lwork);
    int info;
    dgesvd_(
        &jobu, &jobvt, &m, &n, matrix, &m, svd->singular_values.data(),
        svd->left_singular_vectors.data(), &m, svd->right_singular_vectors.data(),
        &n, work_space.data(), &lwork, &info
    );
    assert(info == 0);
    return std::move(svd);
}

TEST_CASE("non square psuedoinverse") {
// >>> A = np.array([[1,2,0],[1,1,1]])
// >>> np.linalg.pinv(A)
// array([[  8.32667268e-17,   3.33333333e-01],
//        [  5.00000000e-01,  -1.66666667e-01],
//        [ -5.00000000e-01,   8.33333333e-01]])
    std::vector<double> matrix{1,2,0,1,1,1};
    auto svd = svd_decompose(matrix.data(), 2, 3);
    auto pseudoinv = svd_pseudoinverse(svd);
    for (int i = 0 ;i < 6; i++) {
        std::cout << pseudoinv[i] << std::endl;
    }
    std::vector<double> correct{0, 1.0 / 3.0, 0.5, -5.0 / 3.0, -0.5, 5.0 / 6.0};
    REQUIRE_ARRAY_CLOSE(pseudoinv, correct, 6, 1e-14); 
}

void set_threshold(const SVDPtr& svd, double threshold) {
    svd->threshold = threshold;
}

int svd_rank(const SVDPtr& svd, double threshold) {
    int rank = 0;
    for (size_t i = 0; i < svd->singular_values.size(); i++) {
        if (svd->singular_values[i] > threshold) {
            rank++;
        }
    }
    return rank;
}

std::vector<double> mat_mult(int n_out_rows, int n_out_cols,
    bool transposeA, std::vector<double>& A,
    bool transposeB, std::vector<double>& B)
{
    std::vector<double> out(n_out_rows * n_out_cols);
    int n_inner = A.size() / n_out_rows;

    //Need to swap from column-major to row-major, so tranposition is inverted.
    char transa = (transposeA) ? 'T' : 'N';
    char transb = (transposeB) ? 'T' : 'N';
    double alpha = 1.0;
    double beta = 0.0;
    // Argument ordering is a little weird for doing row-major
    // instead of fortran-native column-major. See here for details:
    // http://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
    dgemm_(
        &transa, &transb, &n_out_cols, &n_out_rows, &n_inner,
        &alpha, B.data(), &n_out_cols, 
        A.data(), &n_inner,
        &beta, out.data(), &n_out_cols
    );
    return out;
}

TEST_CASE("DGEMM test") {
    std::vector<double> A{0,1,2,3};
    std::vector<double> B{9,8,7,6};
    std::vector<double> correct{7, 6, 39, 34};
    auto result = mat_mult(2, 2, false, A, false, B);
    REQUIRE_ARRAY_EQUAL(result, correct, 4);
}

std::vector<double> svd_pseudoinverse(const SVDPtr& svd) {
    auto n = svd->singular_values.size();
    std::vector<double> left_singular_vectors = svd->left_singular_vectors;

    double cutoff = svd->threshold * svd->singular_values[0];
    for (size_t i = 0; i < n; i++) {
        if (svd->singular_values[i] > cutoff) {
            for (size_t j = 0; j < n; j++) {
                left_singular_vectors[i * n + j] /= svd->singular_values[i];
            }
        } else {
            for (size_t j = 0; j < n; j++) {
                left_singular_vectors[i * n + j] = 0.0;
            }
        }
    }
    
    // left_singular_vectors and right_singular_vectors are both in
    // fortran column-major ordering, so that we can't use the mat_mult function
    // here, which assumes row-major ordering.
    std::vector<double> out(n * n);
    char transa = 'T';
    char transb = 'T';
    int mn = static_cast<int>(n);
    double alpha = 1;
    double beta = 0;
    dgemm_(
        &transa, &transb, &mn, &mn, &mn, &alpha,
        svd->right_singular_vectors.data(), &mn,
        left_singular_vectors.data(), &mn,
        &beta, out.data(), &mn
    );
    return out;
}

std::vector<double> svd_solve(const SVDPtr& svd, const std::vector<double>& b)
{
    // With SVD = U(S)V^T
    // SVD inverse = V(S^{-1})U^T
    // But BLAS input is column-major while my input is row-major, so I want
    // (A^T)^-1 = (V(S^{-1})U^T)^T 
    //          = U(S^{-1})V^T
    auto n = b.size();
    std::vector<double> mult_u_transpose(n, 0.0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            mult_u_transpose[i] += svd->right_singular_vectors[i * n + j] * b[j];
        }
    }

    double cutoff = svd->threshold * svd->singular_values[0];
    for (size_t i = 0; i < n; i++) {
        if (svd->singular_values[i] > cutoff) {
            mult_u_transpose[i] /= svd->singular_values[i];
        } else {
            mult_u_transpose[i] = 0.0;
        }
    }

    std::vector<double> out(n, 0.0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            out[i] += svd->left_singular_vectors[i * n + j] * mult_u_transpose[j];
        }
    }

    return out;
}

double condition_number(const SVDPtr& svd)
{
    auto first = svd->singular_values.front();
    auto last = svd->singular_values.back();
    return first / last;
}

std::vector<double> matrix_vector_product(double* matrix, int n_rows,
    int n_cols, double* vector)
{
    if (n_cols == 0) {
        return {};
    }
    char TRANS = 'T';
    double alpha = 1;
    double beta = 0;
    int inc = 1;
    std::vector<double> out(n_rows);
    //IMPORTANT that n_cols and n_rows are switched because the 3bem internal
    //matrix is in row-major order and BLAS expects column major
    dgemv_(&TRANS, &n_cols, &n_rows, &alpha, matrix,
        &n_cols, vector, &inc, &beta, out.data(), &inc);
    return out;
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

TEST_CASE("Pseudoinverse") 
{
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    auto svd = svd_decompose(matrix.data(), 2, 2);
    auto pseudoinv = svd_pseudoinverse(svd);
    std::vector<double> inv{
        0.25, -0.5, 0.5, 1.0
    };
    REQUIRE_ARRAY_CLOSE(pseudoinv, inv, 4, 1e-14);
}

TEST_CASE("Thresholded pseudoinverse") 
{
    // Matrix has two singular values: 1.0 and 1e-5
    std::vector<double> matrix{
        0.0238032718573239, 0.1524037864980028,
        0.1524037864980028, 0.9762067281426762
    };
    auto svd = svd_decompose(matrix.data(), 2, 2);
    auto no_threshold_pseudoinv = svd_pseudoinverse(svd);
    std::vector<double> correct_no_threshold{
        97620.6728142285282956, -15240.3786497941800917,
        -15240.3786497941782727, 2380.3271857314393856
    };
    REQUIRE_ARRAY_CLOSE(no_threshold_pseudoinv, correct_no_threshold, 4, 1e-4);
    set_threshold(svd, 1e-4);
    auto thresholded_pseudoinv = svd_pseudoinverse(svd);
    std::vector<double> correct_thresholded{
        0.0237935097924219, 0.1524053105511083,
        0.1524053105511085, 0.9762064902075779
    };
    REQUIRE_ARRAY_CLOSE(thresholded_pseudoinv, correct_thresholded, 4, 1e-12);
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

}// end namespace tectosaur
