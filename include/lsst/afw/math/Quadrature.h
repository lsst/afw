// -*- lsst-c++ -*-
#if !defined(LSST_AFW_MATH_QUADRATURE_H)
#define LSST_AFW_MATH_QUADRATURE_H 1

#include <iostream>
#include <functional>
#include <string>
#include <sstream>
#include "boost/format.hpp"
#include "lsst/pex/exceptions.h"

namespace ex = lsst::pex::exceptions;

namespace lsst { namespace afw { namespace math {

/**
 * @file    Quadrature.h
 * @brief   Compute romberg integral and double integral.
 * @ingroup afw/math
 * @author  Steve Bickerton
 * @date    May 25, 2009
 *
 * Outline:
 *
 * The Quadrature::romberg2D() function wraps the function to be integrated in a
 * function object which calls Quadrature::romberg().  romberg2D() then calls romberg()
 * with the wrapped function object.  In essence, romberg() is calling itself
 * recusively to compute the double integral.  To operate this way, the
 * integrand must have a common format, as defined in this IntegrandBase base
 * class.
 *
 */

namespace details {            

// =============================================================
/**
 * @brief Generic functions to get max and min of two numbers.
 */
template<typename T>
inline T max(T a, T b) { return (a >= b) ? a : b; }
template<typename T>
inline T min(T a, T b) { return (a >= b) ? b : a; }

double const epsilon_f = 1.19209e-7;



// =============================================================
/**
 * @class Base_interp
 *
 * @brief A base class for NR intepolation objects.
 * @note Adapted from NR 3rd Ed. Ch 3.
 */
template<typename T>            
class Base_interp {
public:
    
    Base_interp( std::vector<T> &x, std::vector<T> &y, int m)
        : _x(x), _y(y), _m(m) {
        _jsav = 0;
        _n = x.size();
        _cor = 0;
        _dj = min<int>(1, (int) pow((double) _n, 0.25));
    }

    virtual ~Base_interp() {};
    
    T interp(T x) {
        int const jlo = _cor ? hunt(x) : locate(x);
        return rawinterp(jlo, x);
    }
    
    int locate(const T x);
    int hunt(const T x);
    
    T virtual rawinterp(int jlo, T x) = 0;
    
    
protected:
    std::vector<T> &_x;
    std::vector<T> &_y;
    int _n, _m;
private:
    int _jsav, _cor, _dj;
};


// =============================================================
/**
 * @class Poly_interp
 *
 * @brief A Polynomial interpolation object.
 * @note Adapted from NR 3rd Ed. Chap. 3
 */
template <typename T>            
class Poly_interp : public Base_interp<T> {
public:
    Poly_interp(std::vector<T> &xv, std::vector<T>  &yv, int m) :
        Base_interp<T>(xv, yv, m), _dy(0.0) {}
    T rawinterp(int jl, T x);
    T getDy() { return _dy; }
private:
    T _dy;
    using Base_interp<T>::_x;
    using Base_interp<T>::_y;
    using Base_interp<T>::_n;
    using Base_interp<T>::_m;
};




// =============================================================
/**
 * @class Trapzd
 * @brief Perform trapezoid-rule integration.
 * @note Adapted from NR 3d Ed. Chap. 4 pg 163
 */
template<typename UnaryFunctionT>
class Trapzd {
public:
    Trapzd() {};
    Trapzd(UnaryFunctionT func, double a, double b) :
        _func(func), _a(a), _b(b), _s(0) { _n = 0; }
    double next() {
        double x, tnm, sum, del;
        int it, j;
        _n++;
        if (_n == 1) {
            _s = 0.5*(_b - _a)*(_func(_a) + _func(_b));
            return _s;
        } else {
            for (it = 1, j = 1; j < _n - 1; ++j) {
                it <<= 1;
            }
            tnm = it;
            del = (_b - _a)/tnm;
            x = _a + 0.5*del;
            for(sum = 0.0, j = 0; j < it; ++j, x += del) {
                sum += _func(x);
            }
            
            _s = 0.5*(_s + (_b - _a)*sum/tnm);
            return _s;
        }
    }
private:
    UnaryFunctionT _func;
    double _a, _b, _s;
    int _n;
};
    
    
// =============================================================
/**
 * @class FunctionWrapper
 *
 * @brief Wrap an integrand in a call to a 1D integrator: romberg()
 *
 * When romberg2D() is called, it wraps the integrand it was given
 * in a FunctionWrapper functor.  This wrapper calls romberg() on the integrand
 * to get a 1D (along the x-coord, for constant y) result .
 * romberg2D() then calls romberg() with the FunctionWrapper functor as an
 * integrand.
 *
 * @author S. Bickerton (adapted from RHL's SDSS C code)
 */
template<typename BinaryFunctionT>
class FunctionWrapper :
    public std::unary_function<typename BinaryFunctionT::second_argument_type, typename BinaryFunctionT::result_type> {
public:
    FunctionWrapper(BinaryFunctionT func,
                    typename BinaryFunctionT::first_argument_type const x1,
                    typename BinaryFunctionT::first_argument_type const x2,
                    double const eps=1.0e-6) :
        _func(func), _x1(x1), _x2(x2), _eps(eps) {}
    typename BinaryFunctionT::result_type operator() (typename BinaryFunctionT::second_argument_type const y) {
        return romberg(std::bind2nd(_func, y), _x1, _x2, _eps);
    }
private:
    BinaryFunctionT _func;
    typename BinaryFunctionT::first_argument_type _x1, _x2;
    double _eps;
};

} // end of namespace afw::math::details

            
// =============================================================
/**
 * @brief The 1D Romberg integrator
 *
 * @note Adapted from NR 3rd Ed. Chapter 4, pg 166
 *
 * @todo Throw a proper exception when JMAX is exceeded.
 */
template<typename UnaryFunctionT>
typename UnaryFunctionT::result_type romberg(UnaryFunctionT func,
               typename UnaryFunctionT::argument_type const a,
               typename UnaryFunctionT::argument_type const b,
               double eps=1.0e-6)  {

    using namespace details;
    
    static int call_count = 0;
    static int fail_count = 0;
    
    call_count++;
    
    int const JMAX = 20, JMAXP = JMAX + 1, K = 5;
    
    std::vector<typename UnaryFunctionT::argument_type> s(JMAX), h(JMAXP);
    Poly_interp<typename UnaryFunctionT::argument_type> polint(h, s, K);
    h[0] = 1.0;
    Trapzd<UnaryFunctionT> t(func, a, b);
    typename UnaryFunctionT::result_type ss_bail = 0;
    for (int j = 1; j <= JMAX; ++j) {
        s[j - 1] = t.next();
        if (j >= K) {
            typename UnaryFunctionT::result_type ss = ss_bail = polint.rawinterp(j - K, 0.0);
            if ( (std::fabs(polint.getDy()) < std::fabs(eps*ss)) ||
                 (std::fabs(polint.getDy()) < std::fabs(epsilon_f) ) ) {
                return ss;
            }
        }
        h[j] = 0.25*h[j - 1];
    }
    fail_count++;
    
    throw LSST_EXCEPT(ex::RuntimeErrorException,
                      (boost::format("Failed to converge in %d iterations\n") % JMAX).str() );
    return ss_bail; // never gets here.
}


// =============================================================
/**
 * @brief The 2D Romberg integrator
 *
 * @note Adapted from RHL's SDSS code
 *
 */

template<typename BinaryFunctionT>
typename BinaryFunctionT::result_type romberg2D(BinaryFunctionT func,
                                       typename BinaryFunctionT::first_argument_type const x1,
                                       typename BinaryFunctionT::first_argument_type const x2,
                                       typename BinaryFunctionT::second_argument_type const y1,
                                       typename BinaryFunctionT::second_argument_type const y2,
                                       double eps=1.0e-6) {
    using namespace details;
    FunctionWrapper<BinaryFunctionT> fwrap(func, x1, x2, eps);
    return romberg(fwrap, y1, y2, eps);
}

}}}
    
#endif
