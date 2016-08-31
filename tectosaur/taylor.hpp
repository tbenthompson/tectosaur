#ifndef __WUROIWRLKF_TAYLOR_H
#define __WUROIWRLKF_TAYLOR_H
#include <iostream>
#include <array>
#include <cmath>
    
// Citations:
// "Directions for computing truncated multivariate taylor series" by
// Neidinger
//
// "Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation"
// by Andreas Griewank and Andrea Walther. 2008. SIAM.
//
//TODO: Some of the functions can be optimized (look at the two references 
//above)
//TODO: pow fails if c[0] = 0
//TODO: 
//-- sine and cosine
//-- how to get the other operations from sine and cosine and other implemented operations?
//---- use trig definitions (see the papers above)

template <typename T, int M>
struct Taylor {
    static_assert(M >= 1, "Taylor<T,M> should be used with M > 1. \
                 Otherwise it is pointless!");

    std::array<T,M+1> c;
    int n_coeffs;

    Taylor(): c{}, n_coeffs(1) {
        c[0] = 0.0;
    }

    Taylor(const T& x): c{}, n_coeffs(1) {
        c[0] = x;
    }

    T eval(const T& x) {
        T res = 0.0;
        for (int i = n_coeffs - 1; i >= 0; i--) {
            // std::cout << c[i] << std::endl;
            res = c[i] + x * res;
        }
        // std::cout << "DONE" << std::endl;
        return res;
    }

    void sqrt() {
        n_coeffs = M + 1;     
        c[0] = std::sqrt(c[0]);
        for (int i = 1; i < n_coeffs; i++) {
            for (int j = 1; j <= i - 1; j++) {
                c[i] -= c[j] * c[i - j];
            }
            c[i] /= (2 * c[0]);
        }
    }

    void log() {
        n_coeffs = M + 1;     
        std::array<T,M+1> temp_c = c;
        c[0] = std::log(c[0]);
        for (int i = 1; i < n_coeffs; i++) {
            c[i] = i * temp_c[i];
            for (int j = 1; j <= i - 1; j++) {
                c[i] -= j * c[j] * temp_c[i - j];
            }
            c[i] /= (temp_c[0] * i);
        }
    }

    void exp() {
        n_coeffs = M + 1;
        std::array<T,M+1> temp_c = c;
        c[0] = std::exp(c[0]);
        for (int i = 1; i < n_coeffs; i++) {
            c[i] = 0;
            for (int j = 1; j <= i; j++) {
                c[i] += c[i - j] * temp_c[j] * j; 
            }
            c[i] /= i;
        }
    }

    void pow(const T& r) {
        n_coeffs = M + 1;
        std::array<T,M+1> temp_c = c;
        c[0] = std::pow(c[0], r);
        for (int i = 1; i < n_coeffs; i++) {
            double term1 = 0;
            for (int j = 1; j <= i; j++) {
                term1 += c[i - j] * temp_c[j] * j;
            }
            term1 *= r;
            double term2 = 0;
            for (int j = 1; j <= i - 1; j++) {
                term2 -= c[j] * j * temp_c[i - j];
            }
            c[i] = (term1 + term2) / (i * temp_c[0]);
        }
    }


    void operator/=(const Taylor<T,M>& b) {
        n_coeffs = M + 1;
        for (int i = 0; i < n_coeffs; i++) {
            for (int j = 0; j <= i - 1; j++) {
                c[i] -= c[j] * b.c[i - j];
            }
            c[i] /= b.c[0];
        }
    }

    void operator*=(const Taylor<T,M>& b) {
        n_coeffs = std::min(n_coeffs + b.n_coeffs - 1, M + 1);
        std::array<T,M+1> temp_c = c;
        for (int i = 0; i < n_coeffs; i++) {
            c[i] = 0.0;
            for (int j = 0; j <= i; j++) {
                c[i] += temp_c[j] * b.c[i - j];
            }
        }
    }

    void operator+=(const Taylor<T,M>& b) {
        n_coeffs = std::max(n_coeffs, b.n_coeffs);
        for (int i = 0; i < b.n_coeffs; i++) {
            c[i] += b.c[i];
        }
    }

    void operator-=(const Taylor<T,M>& b) {
        (*this) += -b;
    }

    Taylor<T,M> operator-() const {
        Taylor<T,M> res = *this;
        for (int i = 0; i < n_coeffs; i++) {
            res.c[i] = -res.c[i];
        }
        return res;
    }

    Taylor<T,M> operator/(const Taylor<T,M>& b) const {
        Taylor<T,M> res = *this;
        res /= b;
        return res;
    }

    Taylor<T,M> operator*(const Taylor<T,M>& b) const {
        Taylor<T,M> res = *this;
        res *= b;
        return res;
    }
    
    Taylor<T,M> operator+(const Taylor<T,M>& b) const {
        Taylor<T,M> res = *this;
        res += b;
        return res;
    }

    Taylor<T,M> operator+() const {
        return *this;
    }

    Taylor<T,M> operator-(const Taylor<T,M>& b) const {
        Taylor<T,M> res = *this;
        res += -b;
        return res;
    }

    friend std::ostream& operator<<(std::ostream& os, const Taylor<T,M>& t) {
        os << "(";
        for (int i = 0; i < t.n_coeffs - 1; i++) {
            os << t.c[i] << ", ";
        }
        os << t.c[t.n_coeffs - 1];
        os << ")";
        return os;
    }

    static Taylor<T,M> constant(const T& x) {
        return Taylor<T,M>(x);
    }

    static Taylor<T,M> var(const T& x) {
        auto t = Taylor<T,M>(x);
        t.c[1] = 1.0;
        t.n_coeffs = 2;
        return t;
    }
};

template <typename U, typename T, int M>
Taylor<T,M> operator+(const U& x, const Taylor<T,M>& y) {
    return Taylor<T,M>::constant(x) + y;
}

template <typename U, typename T, int M>
Taylor<T,M> operator-(const U& x, const Taylor<T,M>& y) {
    return Taylor<T,M>::constant(x) - y;
}

template <typename U, typename T, int M>
Taylor<T,M> operator*(const U& x, const Taylor<T,M>& y) {
    return Taylor<T,M>::constant(x) * y;
}

template <typename U, typename T, int M>
Taylor<T,M> operator/(const U& x, const Taylor<T,M>& y) {
    return Taylor<T,M>::constant(x) / y;
}

template <typename T, int M>
Taylor<T,M> sqrt(const Taylor<T,M>& t) {
    auto res = t;
    res.sqrt();
    return res;
}

template <typename T, int M>
Taylor<T,M> log(const Taylor<T,M>& t) {
    auto res = t;
    res.log();
    return res;
}

template <typename T, int M>
Taylor<T,M> exp(const Taylor<T,M>& t) {
    auto res = t;
    res.exp();
    return res;
}

template <typename U, typename T, int M>
Taylor<T,M> pow(const Taylor<T,M>& t, const U& r) {
    auto res = t;
    res.pow(r);
    return res;
}

template <int M>
using Td = Taylor<double,M>;

template <int M>
using Tf = Taylor<float,M>;

#endif
