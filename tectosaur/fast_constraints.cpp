<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
%> 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "include/pybind11_nparray.hpp"
#include "include/timing.hpp"

namespace py = pybind11;

struct Term {
    double val;
    size_t dof;

    bool operator==(const Term& b) const {
        return val == b.val && dof == b.dof;
    }
};

using TermVector = std::vector<Term>;

bool operator==(const TermVector& a, const TermVector& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!(a[i] == b[i])) {
            return false;
        }
    }
    return true;
}

struct ConstraintEQ {
    TermVector terms;
    double rhs;

    bool operator==(const ConstraintEQ& b) const {
        return rhs == b.rhs && terms == b.terms;
    }
};

struct ConstructionConstraintEQ {
    TermVector terms;
    TermVector rhs;

    bool operator==(const ConstructionConstraintEQ& b) const {
        return rhs == b.rhs && terms == b.terms;
    }
};

struct IsolatedTermEQ {
    size_t lhs_dof;
    ConstructionConstraintEQ c;

    bool operator==(const IsolatedTermEQ& b) const {
        return lhs_dof == b.lhs_dof && c == b.c;
    }
};

using ConstraintMatrix = std::map<size_t,IsolatedTermEQ>;

void print_c(const ConstructionConstraintEQ& c, bool recurse, ConstraintMatrix lower_tri_cs) {
    for (size_t i = 0; i < c.rhs.size(); i++) {
        std::cout << " rhs(" << i << "):" << c.rhs[i].dof << " " << c.rhs[i].val << std::endl;
    }
    for (size_t i = 0; i < c.terms.size(); i++) {
        std::cout << " term(" << i << "):" << c.terms[i].dof << " " << c.terms[i].val << std::endl;
        if (recurse && lower_tri_cs.count(c.terms[i].dof) != 0) {
            print_c(lower_tri_cs[c.terms[i].dof].c, false, lower_tri_cs);
        }
    }
}

IsolatedTermEQ isolate_term_on_lhs(const ConstructionConstraintEQ& c, size_t entry_idx) {
    auto lhs = c.terms[entry_idx];

    TermVector divided_negated_terms;
    for (size_t i = 0; i < c.terms.size(); i++) {
        if (i == entry_idx) {
            continue;
        }
        divided_negated_terms.push_back({-c.terms[i].val / lhs.val, c.terms[i].dof});
    }
    TermVector divided_rhs;
    for (size_t i = 0; i < c.rhs.size(); i++) {
        divided_rhs.push_back({c.rhs[i].val / lhs.val, c.rhs[i].dof});
    }
    return IsolatedTermEQ{lhs.dof, divided_negated_terms, divided_rhs};
}

ConstructionConstraintEQ substitute(const ConstructionConstraintEQ& c_victim, size_t entry_idx,
    const IsolatedTermEQ& c_in, double rhs_factor) 
{
    assert(c_in.lhs_dof == c_victim.terms[entry_idx].dof);
    double mult_factor = c_victim.terms[entry_idx].val;

    TermVector out_terms;
    for (size_t i = 0; i < c_victim.terms.size(); i++) {
        if (i == entry_idx) {
            continue;
        }
        out_terms.push_back(c_victim.terms[i]);
    }

    for (size_t i = 0; i < c_in.c.terms.size(); i++) {
        out_terms.push_back({c_in.c.terms[i].val * mult_factor, c_in.c.terms[i].dof});
    }

    TermVector out_rhs = c_victim.rhs;
    for (size_t i = 0; i < c_in.c.rhs.size(); i++) {
        out_rhs.push_back({-rhs_factor * mult_factor * c_in.c.rhs[i].val, c_in.c.rhs[i].dof});
    }
    ConstructionConstraintEQ out{out_terms, out_rhs};
    return out;
}

TermVector combine_terms_helper(const TermVector& tv) {
    TermVector out_terms;
    for (size_t i = 0; i < tv.size(); i++) {
        auto& t = tv[i];
        bool found = false;
        for (size_t j = 0; j < out_terms.size(); j++) {
            if (out_terms[j].dof == t.dof) {
                out_terms[j].val += t.val;
                found = true;
                break;
            }
        }
        if (!found) {
            out_terms.push_back(t);
        }
    }
    return out_terms;
}

ConstructionConstraintEQ combine_terms(const ConstructionConstraintEQ& c) {
    return ConstructionConstraintEQ{
        combine_terms_helper(c.terms),
        combine_terms_helper(c.rhs)
    };
}

TermVector filter_zero_helper(const TermVector& tv) {
    TermVector out_terms;
    for (size_t i = 0; i < tv.size(); i++) {
        if (std::fabs(tv[i].val) < 1e-15) {
            continue;
        }
        out_terms.push_back(tv[i]);
    }
    return out_terms;
}

ConstructionConstraintEQ filter_zero_terms(const ConstructionConstraintEQ& c) {
    return ConstructionConstraintEQ{
        filter_zero_helper(c.terms),
        filter_zero_helper(c.rhs)
    };
}

std::pair<size_t,size_t> max_dof(const TermVector& terms) {
    size_t max_dof = 0;
    size_t idx = 0;
    for (size_t i = 0; i < terms.size(); i++) {
        if (terms[i].dof > max_dof) {
            max_dof = terms[i].dof;
            idx = i;
        }
    }
    return std::make_pair(max_dof, idx);
}

ConstructionConstraintEQ make_reduced(const ConstructionConstraintEQ& c, const ConstraintMatrix& m, double rhs_factor) {
    for (size_t i = 0; i < c.terms.size(); i++) {
        if (m.count(c.terms[i].dof) > 0) {
            auto existing_c = m.at(c.terms[i].dof);
            auto c_subs = substitute(c, i, existing_c, rhs_factor);
            auto c_combined = combine_terms(c_subs);
            auto c_filtered = filter_zero_terms(c_combined);
            return make_reduced(c_filtered, m, rhs_factor);
        }
    }
    return c;
}

ConstructionConstraintEQ to_construction(const ConstraintEQ& c, size_t i) {
    ConstructionConstraintEQ out;
    out.terms = c.terms;
    out.rhs = {Term{1.0, i}};
    return out;
}

ConstraintMatrix reduce_constraints(std::vector<ConstraintEQ> cs,
    size_t n_total_dofs) 
{
    std::vector<size_t> order(cs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(), order.end(), 
        [&] (size_t a_i, size_t b_i) {
            auto a = cs[a_i];
            auto b = cs[b_i];
            if (a.terms.size() == b.terms.size()) {
                return max_dof(a.terms).first < max_dof(b.terms).first;
            }
            return a.terms.size() < b.terms.size();
        }
    );

    ConstraintMatrix lower_tri_cs;
    for (size_t o_i = 0; o_i < order.size(); o_i++) {
        size_t i = order[o_i];
        auto c = to_construction(cs[i], i);
        auto c_combined = combine_terms(c);
        auto c_filtered = filter_zero_terms(c_combined);
        auto c_lower_tri = make_reduced(c_filtered, lower_tri_cs, 1.0);
        
        if (c_lower_tri.terms.size() == 0) {
            continue;
        }
        
        auto ldi = max_dof(c_lower_tri.terms).second;
        auto separated = isolate_term_on_lhs(c_lower_tri, ldi);
        lower_tri_cs[separated.lhs_dof] = separated;
    }

    for (size_t i = 0; i < n_total_dofs; i++) {
        if (lower_tri_cs.count(i) == 0) {
            continue;
        }

        auto before = lower_tri_cs[i].c;
        lower_tri_cs[i].c = make_reduced(lower_tri_cs[i].c, lower_tri_cs, -1.0);
    }
    
    return lower_tri_cs;
}

py::tuple build_constraint_matrix(const std::vector<ConstraintEQ>& cs, size_t n_total_dofs) {

    Timer t(true);
    auto lower_tri_cs = reduce_constraints(cs, n_total_dofs);
    t.report("reduce");

    std::vector<size_t> rows;    
    std::vector<size_t> cols;    
    std::vector<double> vals;

    std::vector<size_t> rhs_rows;    
    std::vector<size_t> rhs_cols;    
    std::vector<double> rhs_vals;

    std::vector<double> rhs_input(cs.size(), 0.0);
    for (size_t i = 0; i < cs.size(); i++) {
        rhs_input[i] = cs[i].rhs; 
    }

    size_t next_new_dof = 0;
    std::map<size_t,size_t> new_dofs;
    for (size_t i = 0; i < n_total_dofs; i++) {
        if (lower_tri_cs.count(i) > 0) {
            for (auto& t: lower_tri_cs[i].c.terms) {
                assert(new_dofs.count(t.dof) > 0);
                rows.push_back(i);
                cols.push_back(new_dofs[t.dof]);
                vals.push_back(t.val);
            }
            for (auto& t: lower_tri_cs[i].c.rhs) {
                rhs_rows.push_back(i);
                rhs_cols.push_back(t.dof);
                rhs_vals.push_back(t.val);
            }
        } else {
            rows.push_back(i);
            cols.push_back(next_new_dof);
            vals.push_back(1);
            new_dofs[i] = next_new_dof;
            next_new_dof++;
        }
    }
    t.report("insert");
    auto out = py::make_tuple(
        array_from_vector(rows),
        array_from_vector(cols),
        array_from_vector(vals),
        array_from_vector(rhs_rows),
        array_from_vector(rhs_cols),
        array_from_vector(rhs_vals),
        array_from_vector(rhs_input),
        lower_tri_cs.size()
    );
    t.report("make out");
    return out;
}


PYBIND11_MODULE(fast_constraints,m) {
    py::class_<Term>(m, "Term")
        .def(py::init([] (double val, size_t dof) {
            return Term{val, dof};
        }))
        .def_readonly("val", &Term::val)
        .def_readonly("dof", &Term::dof);

    py::class_<ConstraintEQ>(m, "ConstraintEQ")
        .def(py::init([] (const TermVector& terms, double rhs) {
            return ConstraintEQ{terms, rhs};
        }))
        .def_readonly("terms", &ConstraintEQ::terms)
        .def_readonly("rhs", &ConstraintEQ::rhs)
        .def(py::self == py::self);

    py::class_<ConstructionConstraintEQ>(m, "ConstructionConstraintEQ")
        .def_readonly("terms", &ConstructionConstraintEQ::terms)
        .def_readonly("rhs", &ConstructionConstraintEQ::rhs)
        .def(py::self == py::self);

    py::class_<IsolatedTermEQ>(m, "IsolatedTermEQ")
        .def_readonly("lhs_dof", &IsolatedTermEQ::lhs_dof)
        .def_readonly("c", &IsolatedTermEQ::c)
        .def(py::self == py::self);

    m.def("isolate_term_on_lhs", [&] (const ConstraintEQ& c, size_t entry_idx) {
        return isolate_term_on_lhs(to_construction(c, 0), entry_idx);
    });
    m.def("substitute", [&] (const ConstraintEQ& c_victim, size_t entry_idx,
            const IsolatedTermEQ& c_in, double rhs_factor) {
        return substitute(to_construction(c_victim, 0), entry_idx, c_in, rhs_factor);
    });
    m.def("combine_terms", [&] (const ConstraintEQ& c) {
        return combine_terms(to_construction(c, 0));
    });
    m.def("filter_zero_terms", [&] (const ConstraintEQ& c) {
        return filter_zero_terms(to_construction(c, 0));
    });
    m.def("build_constraint_matrix", &build_constraint_matrix);
}
