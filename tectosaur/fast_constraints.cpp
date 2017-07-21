<%
setup_pybind11(cfg)
%> 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

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
    TermVector terms;
    double rhs;

    bool operator==(const IsolatedTermEQ& b) const {
        return lhs_dof == b.lhs_dof && rhs == b.rhs && terms == b.terms;
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

    for (size_t i = 0; i < c_in.terms.size(); i++) {
        out_terms.push_back({c_in.terms[i].val * mult_factor, c_in.terms[i].dof});
    }

    double out_rhs = c_victim.rhs - mult_factor * c_in.rhs;
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

PYBIND11_PLUGIN(fast_constraints) {
    py::module m("fast_constraints", "");

    py::class_<Term>(m, "Term")
        .def("__init__", [] (Term& t, double val, size_t dof) {
            t.val = val;
            t.dof = dof; 
        })
        .def_readonly("val", &Term::val)
        .def_readonly("dof", &Term::dof);

    py::class_<ConstraintEQ>(m, "ConstraintEQ")
        .def("__init__", [] (ConstraintEQ& c, const TermVector& terms, double rhs) {
            new (&c) ConstraintEQ{terms, rhs};
        })
        .def_readonly("terms", &ConstraintEQ::terms)
        .def_readonly("rhs", &ConstraintEQ::rhs)
        .def(py::self == py::self);

    py::class_<IsolatedTermEQ>(m, "IsolatedTermEQ")
        .def_readonly("lhs_dof", &IsolatedTermEQ::lhs_dof)
        .def_readonly("terms", &IsolatedTermEQ::terms)
        .def_readonly("rhs", &IsolatedTermEQ::rhs)
        .def(py::self == py::self);

    m.def("isolate_term_on_lhs", &isolate_term_on_lhs);
    m.def("substitute", &substitute);

    return m.ptr();
}
