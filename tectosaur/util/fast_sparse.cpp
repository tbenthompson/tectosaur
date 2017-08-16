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

<%def name="bsrmv(blocksize)">
template <typename I, typename F>
void bsrmv${blocksize}(size_t mb, NPArray<I> indptr, NPArray<I> indices,
        NPArray<F> data, NPArray<F> x, NPArray<F> y) 
{
    auto* indptr_ptr = as_ptr<I>(indptr);
    auto* indices_ptr = as_ptr<I>(indices);
    auto* A_ptr = as_ptr<F>(data);
    auto* x_ptr = as_ptr<F>(x);
    auto* y_ptr = as_ptr<F>(y);

#pragma omp parallel for
    for (size_t block_row_idx = 0; block_row_idx < mb; block_row_idx++) {
        auto* y_start = y_ptr + ${blocksize} * block_row_idx;

        % for un_i in range(blocksize):
        F sum${un_i} = 0.0;
        % endfor

        I col_ptr_start = indptr_ptr[block_row_idx];
        I col_ptr_end = indptr_ptr[block_row_idx + 1];
        for (I col_ptr = col_ptr_start;
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

% for blocksize in range(1, 10):
    ${bsrmv(blocksize)}
% endfor 

PYBIND11_PLUGIN(fast_sparse) {
    py::module m("fast_sparse");
    % for blocksize in range(1, 10):
        m.def("sbsrmv${blocksize}", &bsrmv${blocksize}<int,float>);
        m.def("dbsrmv${blocksize}", &bsrmv${blocksize}<long,double>);
    % endfor
    return m.ptr();
}

