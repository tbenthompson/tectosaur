#pragma once

#include <vector>
#include <memory>

namespace tectosaur {

struct LU;
struct LUDeleter {
    void operator()(LU* thing);
};
typedef std::unique_ptr<LU,LUDeleter> LUPtr;

LUPtr lu_decompose(const std::vector<double>& matrix);
std::vector<double> lu_solve(const LUPtr& lu, const std::vector<double>& b);

struct SVD;
struct SVDDeleter {
    void operator()(SVD* thing);
};
typedef std::unique_ptr<SVD,SVDDeleter> SVDPtr;

std::vector<double> qr_pseudoinverse(double* matrix, int m, int n);
SVDPtr svd_decompose(double* matrix, int n);
void set_threshold(const SVDPtr& svd, double threshold);
std::vector<double> svd_pseudoinverse(const SVDPtr& svd);
std::vector<double> svd_solve(const SVDPtr& svd, const std::vector<double>& b);
double condition_number(const SVDPtr& svd); 

std::vector<double> mat_mult(int n_out_rows, int n_out_cols,
    bool transposeA, std::vector<double>& A,
    bool transposeB, std::vector<double>& B);
std::vector<double> matrix_vector_product(double* matrix, int n_rows,
    int n_cols, double* vector);

} // end namespace tectosaur
