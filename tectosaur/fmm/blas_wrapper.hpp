#pragma once

#include <vector>
#include <memory>

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
SVDPtr svd_decompose(double* matrix, int m, int n);
void set_threshold(const SVDPtr& svd, double threshold);
int svd_rank(const SVDPtr& svd, double threshold);
std::vector<double> svd_solve(const SVDPtr& svd, const std::vector<double>& b);
double condition_number(const SVDPtr& svd); 

std::vector<double> mat_mult(int n_out_rows, int n_out_cols,
    bool transposeA, std::vector<double>& A,
    bool transposeB, std::vector<double>& B);
void matrix_vector_product(double* matrix, int n_rows, int n_cols,
    double* vector, double* out);
std::vector<double> matrix_vector_product(double* matrix, int n_rows,
    int n_cols, double* vector);

struct Block {
    size_t row_start;
    size_t col_start;
    int n_rows;
    int n_cols;
    size_t data_start;
};

struct BlockSparseMat {
    std::vector<Block> blocks;
    std::vector<double> vals;

    std::vector<double> matvec(double* vec, size_t out_size);
    size_t get_nnz() { return vals.size(); }
};

std::vector<double> qr_pseudoinverse(double* matrix, int n, double cond_cutoff);
