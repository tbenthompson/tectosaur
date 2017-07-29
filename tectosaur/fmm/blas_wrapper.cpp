#include "blas_wrapper.hpp"
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

void matrix_vector_product(double* matrix, int n_rows, int n_cols,
    double* vector, double* out) 
{
    if (n_cols == 0) {
        return;
    }
    char TRANS = 'T';
    double alpha = 1.0;
    double beta = 1.0;
    int inc = 1.0;
    //IMPORTANT that n_cols and n_rows are switched because the 3bem internal
    //matrix is in row-major order and BLAS expects column major
    dgemv_(&TRANS, &n_cols, &n_rows, &alpha, matrix,
        &n_cols, vector, &inc, &beta, out, &inc);
}

std::vector<double> matrix_vector_product(double* matrix, int n_rows,
    int n_cols, double* vector)
{
    std::vector<double> out(n_rows, 0.0);
    matrix_vector_product(matrix, n_rows, n_cols, vector, out.data());
    return out;
}

extern "C" void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A,
                       int* LDA, double* X, int* INCX, double* BETA, double* Y,
                       int* INCY);
std::vector<double> BlockSparseMat::matvec(double* vec, size_t out_size) {
    char transpose = 'T';
    double alpha = 1;
    double beta = 1;
    int inc = 1;
    std::vector<double> out(out_size, 0.0);
    for (size_t b_idx = 0; b_idx < blocks.size(); b_idx++) {
        auto& b = blocks[b_idx];
        dgemv_(
            &transpose, &b.n_cols, &b.n_rows, &alpha, &vals[b.data_start],
            &b.n_cols, &vec[b.col_start], &inc, &beta, &out[b.row_start], &inc
        );
    }
    return out;
}

extern "C" void dgelsy_(int* M, int* N, int* NRHS, double* A, int* LDA,
                        double* B, int* LDB, int* JPVT, double* RCOND,
                        int* RANK, double* WORK, int* LWORK, int* INFO);

std::vector<double> qr_pseudoinverse(double* matrix, int n, double cond_cutoff) {
    std::vector<double> rhs(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        rhs[i * n + i] = 1.0;
    }

    std::vector<int> jpvt(n, 0);
    int lwork = 4 * n + 1;
    std::vector<double> work(lwork);
    int rank;
    int info;
    dgelsy_(&n, &n, &n, matrix, &n, rhs.data(), &n, jpvt.data(), &cond_cutoff, &rank,
            work.data(), &lwork, &info);
    return rhs;
}


