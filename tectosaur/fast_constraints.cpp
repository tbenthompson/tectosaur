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

struct IsolatedTermEQ {
    size_t lhs_dof;
    ConstraintEQ c;

    bool operator==(const IsolatedTermEQ& b) const {
        return lhs_dof == b.lhs_dof && c == b.c;
    }
};

IsolatedTermEQ isolate_term_on_lhs(const ConstraintEQ& c, size_t entry_idx) {
    auto lhs = c.terms[entry_idx];

    TermVector divided_negated_terms;
    for (size_t i = 0; i < c.terms.size(); i++) {
        if (i == entry_idx) {
            continue;
        }
        divided_negated_terms.push_back({-c.terms[i].val / lhs.val, c.terms[i].dof});
    }
    double divided_rhs = c.rhs / lhs.val;
    return IsolatedTermEQ{lhs.dof, divided_negated_terms, divided_rhs};
}

ConstraintEQ substitute(const ConstraintEQ& c_victim, size_t entry_idx,
    const IsolatedTermEQ& c_in) 
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

    double out_rhs = c_victim.rhs + mult_factor * c_in.c.rhs;
    return ConstraintEQ{out_terms, out_rhs};
}

ConstraintEQ combine_terms(const ConstraintEQ& c) {
    TermVector out_terms;
    for (size_t i = 0; i < c.terms.size(); i++) {
        auto& t = c.terms[i];
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
    return ConstraintEQ{out_terms, c.rhs};
}

ConstraintEQ filter_zero_terms(const ConstraintEQ& c) {
    TermVector out_terms;
    for (size_t i = 0; i < c.terms.size(); i++) {
        if (std::fabs(c.terms[i].val) < 1e-15) {
            continue;
        }
        out_terms.push_back(c.terms[i]);
    }
    return ConstraintEQ{out_terms, c.rhs};
}

std::pair<size_t,size_t> max_dof(const ConstraintEQ& c) {
    size_t max_dof = 0;
    size_t idx = 0;
    for (size_t i = 0; i < c.terms.size(); i++) {
        if (c.terms[i].dof > max_dof) {
            max_dof = c.terms[i].dof;
            idx = i;
        }
    }
    return std::make_pair(max_dof, idx);
}

using ConstraintMatrix = std::map<size_t,IsolatedTermEQ>;

ConstraintEQ make_reduced(const ConstraintEQ& c, const ConstraintMatrix& m) {
    for (size_t i = 0; i < c.terms.size(); i++) {
        if (m.count(c.terms[i].dof) > 0) {
            auto c_subs = substitute(c, i, m.at(c.terms[i].dof));
            auto c_combined = combine_terms(c_subs);
            auto c_filtered = filter_zero_terms(c_combined);
            return make_reduced(c_filtered, m);
        }
    }
    return c;
}

void print_c(const ConstraintEQ& c, bool recurse, ConstraintMatrix lower_tri_cs) {
    std::cout << "rhs: " << c.rhs << std::endl;
    for (size_t i = 0; i < c.terms.size(); i++) {
        std::cout << " term(" << i << "):" << c.terms[i].dof << " " << c.terms[i].val << std::endl;
        if (recurse && lower_tri_cs.count(c.terms[i].dof) != 0) {
            print_c(lower_tri_cs[c.terms[i].dof].c, false, lower_tri_cs);
        }
    }
}


ConstraintMatrix reduce_constraints(std::vector<ConstraintEQ> cs,
    size_t n_total_dofs) 
{
    std::sort(
        cs.begin(), cs.end(), 
        [] (const ConstraintEQ& a, const ConstraintEQ& b) {
            return max_dof(a).first < max_dof(b).first;
        }
    );

    ConstraintMatrix lower_tri_cs;
    for (size_t i = 0; i < cs.size(); i++) {
        auto c = cs[i];
        auto c_combined = combine_terms(c);
        auto c_filtered = filter_zero_terms(c_combined);
        auto c_lower_tri = make_reduced(c_filtered, lower_tri_cs);
        
        if (c_lower_tri.terms.size() == 0) {
            continue;
        }
        
        auto ldi = max_dof(c_lower_tri).second;
        auto separated = isolate_term_on_lhs(c_lower_tri, ldi);
        lower_tri_cs[separated.lhs_dof] = separated;
        if (separated.c.rhs < 0) {
            std::cout << "YIKES!" << std::endl;
        }
    }

    for (size_t i = 0; i < n_total_dofs; i++) {
        if (lower_tri_cs.count(i) == 0) {
            continue;
        }

        auto before = lower_tri_cs[i].c;
        lower_tri_cs[i].c = make_reduced(lower_tri_cs[i].c, lower_tri_cs);
        if (lower_tri_cs[i].c.rhs < 0) {
            std::cout << "YIKES2!" << std::endl;
            std::cout << "constrained_dof: " << i << std::endl;
            std::cout << "before: ";print_c(before, true, lower_tri_cs);
            std::cout << "after: ";print_c(lower_tri_cs[i].c, true, lower_tri_cs);
            std::cout << std::endl;
        }
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
    std::vector<double> rhs(n_total_dofs, 0.0);
    size_t next_new_dof = 0;
    std::map<size_t,size_t> new_dofs;
    for (size_t i = 0; i < n_total_dofs; i++) {
        if (lower_tri_cs.count(i) > 0) {
            rhs[i] = lower_tri_cs[i].c.rhs;
            for (auto& t: lower_tri_cs[i].c.terms) {
                assert(new_dofs.count(t.dof) > 0);
                rows.push_back(i);
                cols.push_back(new_dofs[t.dof]);
                vals.push_back(t.val);
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
        array_from_vector(rhs),
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

    py::class_<IsolatedTermEQ>(m, "IsolatedTermEQ")
        .def_readonly("lhs_dof", &IsolatedTermEQ::lhs_dof)
        .def_readonly("c", &IsolatedTermEQ::c)
        .def(py::self == py::self);

    m.def("isolate_term_on_lhs", &isolate_term_on_lhs);
    m.def("substitute", &substitute);
    m.def("combine_terms", &combine_terms);
    m.def("filter_zero_terms", &filter_zero_terms);
    m.def("build_constraint_matrix", &build_constraint_matrix);
}
