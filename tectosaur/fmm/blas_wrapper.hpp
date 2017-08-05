#pragma once

#include <vector>
#include <memory>

void matrix_vector_product(double* matrix, int n_rows, int n_cols,
    double* vector, double* out);

std::vector<double> qr_pseudoinverse(double* matrix, int n, double cond_cutoff);
