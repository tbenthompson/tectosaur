<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'] = [
    '../include/pybind11_nparray.hpp',
]
%>

#include <pybind11/pybind11.h>
#include "include/pybind11_nparray.hpp"

namespace py = pybind11;

py::tuple make_bsr_matrix(size_t n_rows, size_t n_cols, 
        NPArray<double> in_data, NPArray<long> rows, NPArray<long> cols) 
{
    size_t blockrows = in_data.request().shape[1];
    size_t blockcols = in_data.request().shape[2];
    assert(blockrows == blockcols);

    size_t n_row_blocks = n_rows / blockrows;
    size_t n_blocks = rows.request().shape[0];
    size_t blocksize = blockrows * blockcols;

    auto* rows_ptr = as_ptr<long>(rows);
    auto* cols_ptr = as_ptr<long>(cols);
    auto* in_data_ptr = as_ptr<double>(in_data);

    auto indptr = make_array<long>({n_row_blocks + 1});
    auto indices = make_array<long>({n_blocks});
    auto data = make_array<double>({n_blocks, blockrows, blockcols});

    auto* indptr_ptr = as_ptr<long>(indptr);
    auto* indices_ptr = as_ptr<long>(indices);
    auto* data_ptr = as_ptr<double>(data);

    std::fill(indptr_ptr, indptr_ptr + n_row_blocks, 0.0);

    for (size_t i = 0; i < n_blocks; i++) {
        indptr_ptr[rows_ptr[i]]++;
    }

    for (size_t i = 0, cumsum = 0; i < n_row_blocks; i++) {
        int temp = indptr_ptr[i];
        indptr_ptr[i] = cumsum;
        cumsum += temp;
    }
    indptr_ptr[n_row_blocks] = n_blocks;

    for (size_t n = 0; n < n_blocks; n++) {
        int row = rows_ptr[n];
        int dest = indptr_ptr[row];

        indices_ptr[dest] = cols_ptr[n];
        for (size_t k = 0; k < blocksize; k++) {
            data_ptr[dest * blocksize + k] = in_data_ptr[n * blocksize + k];
        }

        indptr_ptr[row]++;
    }

    for (size_t i = 0, last = 0; i <= n_row_blocks; i++) {
        int temp = indptr_ptr[i];
        indptr_ptr[i] = last;
        last = temp;
    }
    return py::make_tuple(indptr, indices, data);
}


<%def name="bsrmv(blocksize)">
template <typename F>
void bsrmv${blocksize}(NPArray<long> indptr, NPArray<long> indices,
        NPArray<F> data, NPArray<F> x, NPArray<F> y) 
{
    size_t mb = indptr.request().shape[0] - 1;

    auto* indptr_ptr = as_ptr<long>(indptr);
    auto* indices_ptr = as_ptr<long>(indices);
    auto* A_ptr = as_ptr<F>(data);
    auto* x_ptr = as_ptr<F>(x);
    auto* y_ptr = as_ptr<F>(y);

#pragma omp parallel for
    for (size_t block_row_idx = 0; block_row_idx < mb; block_row_idx++) {
        auto* y_start = y_ptr + ${blocksize} * block_row_idx;

        % for un_i in range(blocksize):
        F sum${un_i} = 0.0;
        % endfor

        long col_ptr_start = indptr_ptr[block_row_idx];
        long col_ptr_end = indptr_ptr[block_row_idx + 1];
        for (long col_ptr = col_ptr_start;
                col_ptr < col_ptr_end;
                col_ptr++) 
        {
            auto block_col_idx = indices_ptr[col_ptr];
            auto* A_start = A_ptr + ${blocksize ** 2} * col_ptr;
            auto* x_start = x_ptr + ${blocksize} * block_col_idx;
            % for un_j in range(blocksize):
                auto x${un_j} = x_start[${un_j}];
            % endfor
            % for un_i in range(blocksize):
                sum${un_i} += 
                % for un_j in range(blocksize):
                    + A_start[${un_i * blocksize + un_j}] * x${un_j}
                % endfor
                ;
            % endfor
        }
        % for un_i in range(blocksize):
        y_start[${un_i}] = sum${un_i};
        % endfor
    }
}
</%def>

#define CACHE_LINE_SIZE 64
#define PrefetchRange(a, n, rw, t) do {                               \
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(a);\
    const uint8_t* end = reinterpret_cast<const uint8_t*>((a)+(n));   \
    for (; ptr < end; ptr += CACHE_LINE_SIZE) {\
        __builtin_prefetch(ptr,(rw),(t));\
    }\
  } while (0)

<%def name="bcoomv(blocksize)">
template <typename F>
void bcoomv${blocksize}(NPArray<long> rows, NPArray<long> cols,
        NPArray<F> data, NPArray<F> x, NPArray<F> y) 
{
    size_t n_blocks = rows.request().shape[0];

    auto* rows_ptr = as_ptr<long>(rows);
    auto* cols_ptr = as_ptr<long>(cols);
    auto* A_ptr = as_ptr<F>(data);
    auto* x_ptr = as_ptr<F>(x);
    auto* y_ptr = as_ptr<F>(y);

#pragma omp parallel for
    for (size_t block_idx = 0; block_idx < n_blocks; block_idx++) {
        auto* x_start = x_ptr + ${blocksize} * cols_ptr[block_idx];
        auto* y_start = y_ptr + ${blocksize} * rows_ptr[block_idx];
        auto* A_start = A_ptr + ${blocksize ** 2} * block_idx;
        PrefetchRange(x_ptr + ${blocksize} * cols_ptr[block_idx + 1],${blocksize},0,0);
        PrefetchRange(y_ptr + ${blocksize} * rows_ptr[block_idx + 1],${blocksize},1,0);

        % for un_j in range(blocksize):
            auto x${un_j} = x_start[${un_j}];
        % endfor

        % for un_i in range(blocksize):
#pragma omp atomic
            y_start[${un_i}] += 
            % for un_j in range(blocksize):
                + A_start[${un_i * blocksize + un_j}] * x${un_j}
            % endfor
            ;
        % endfor
    }
}
</%def>

% for blocksize in range(1, 10):
    ${bsrmv(blocksize)}
    ${bcoomv(blocksize)}
% endfor 

PYBIND11_MODULE(fast_sparse,m) {
    m.def("make_bsr_matrix", &make_bsr_matrix);
    % for blocksize in range(1, 10):
        m.def("sbsrmv${blocksize}", &bsrmv${blocksize}<float>);
        m.def("dbsrmv${blocksize}", &bsrmv${blocksize}<double>);
        m.def("sbcoomv${blocksize}", &bcoomv${blocksize}<float>);
        m.def("dbcoomv${blocksize}", &bcoomv${blocksize}<double>);
    % endfor
}

