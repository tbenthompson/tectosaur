#include "blas_wrapper.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

extern "C" void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A,
    int* LDA, double* X, int* INCX, double* BETA, double* Y, int* INCY);

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


