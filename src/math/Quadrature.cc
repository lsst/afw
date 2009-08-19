// -*- LSST-C++ -*-
/**
 * @file    Quadrature.cc
 * @ingroup afw
 * @brief   Compute 1- and 2-D integrals with Romberg integration.
 * @author  Steve Bickerton
 * @date    May 25, 2009
 */
#include <vector>
#include <cstdlib>
#include <cmath>

#include "lsst/afw/math/Quadrature.h"

namespace math = lsst::afw::math;

/**
 * Almost all of the classes/functions in this anonymous namespace
 * are adapted from Numerical Recipes 3'rd Ed.
 */
    

namespace lsst { namespace afw { namespace math {

using namespace details;
            
/**
 * @brief See NR 3rd pg. 115.
 *
 */
template <typename T>
int Base_interp<T>::locate(T const x) {
    
    int ju, jm, jl;
    if ( _n < 2 || _m < 2 || _m > _n ) {
        throw("locate size error");
    }
    bool ascnd = (_x[_n - 1] >= _x[0]);
    jl = 0;
    ju = _n - 1;
    while (ju - jl > 1) {
        jm = (ju + jl) >> 1;
        if ( (x >= _x[jm]) == ascnd ) {
            jl = jm;
        } else {
            ju = jm;
        }
    }
    _cor = (std::abs(jl - _jsav) > _dj) ? 0 : 1;
    _jsav = jl;
    return max<int>(0, min<int>(_n - _m, jl - ((_m - 2)>>1) ) );
}

/**
 * @brief See NR 3rd pg. 116.
 *
 */
template <typename T>
int Base_interp<T>::hunt(T const x) {
    
    int jl = _jsav, jm, ju, inc = 1;
    if (_n < 2 || _m < 2 || _m > _n) { throw ("hunt size error"); }
    bool ascnd = (_x[_n - 1] >= _x[0]);
    if (jl < 0 || jl > _n - 1) {
        jl = 0;
        ju = _n - 1;
    } else {
        if ( (x >= _x[jl]) == ascnd) {
            for (;;) {
                ju = jl + inc;
                if (ju >= _n - 1 ) {
                    ju = _n - 1;
                    break;
                } else if ( (x < _x[ju]) == ascnd) {
                    break;
                } else {
                    jl = ju;
                    inc += inc;
                }
            }
        } else {
            ju = jl;
            for (;;) {
                jl = jl - inc;
                if (jl <= 0 ) {
                    jl = 0;
                    break; 
                } else if ( (x >= _x[jl]) == ascnd) {
                    break; 
                } else {
                    ju = jl;
                    inc += inc;
                }
            }
        }
    }
    while (ju - jl > 1) {
        jm = (ju + jl) >> 1;
        if ( (x >= _x[jm]) == ascnd) {
            jl = jm;
        } else {
            ju = jm;
        }
    }
    _cor = std::abs(jl - _jsav) > _dj ? 0 : 1;
    _jsav = jl;
    return max<int>(0, min<int>(_n - _m, jl - ((_m - 2) >> 1)) );
}


/**
 * @brief See NR 3rd pg. 119.
 *
 */
template <typename T>
T Poly_interp<T>::rawinterp(int const jl, T const x) {
    int i, m, ns = 0;
    double y, den, dif, dift, ho, hp, w;
    typename std::vector<T>::pointer xa = &_x[jl];
    typename std::vector<T>::pointer ya = &_y[jl];
    std::vector<T> c(_m), d(_m);
    dif = std::fabs(x - xa[0]);
    for (i = 0; i < _m; ++i) {
        if ((dift = std::fabs(x - xa[i])) < dif) {
            ns = i;
            dif = dift;
        }
        c[i] = ya[i];
        d[i] = ya[i];
    }
    y = ya[ns--];
    for (m = 1; m < _m; ++m) {
        for (i = 0; i < _m - m; ++i ) {
            ho = xa[i] - x;
            hp = xa[i+m] - x;
            w = c[i+1] - d[i];
            if ((den = ho - hp) == 0.0) {
                throw("Poly_interp error");
            }
            den = w/den;
            d[i] = hp*den;
            c[i] = ho*den;
        }
        y += (_dy = (2*(ns+1) < (_m - m) ? c[ns+1] : d[ns--]));    
    }
    return y;
}

            
#define INSTANTIATE_QUADRATURE(TYPE) \
            template int Base_interp<TYPE>::locate(const TYPE); \
            template int Base_interp<TYPE>::hunt(const TYPE);   \
            template TYPE Poly_interp<TYPE>::rawinterp(int const, TYPE const);         

INSTANTIATE_QUADRATURE(double);            
INSTANTIATE_QUADRATURE(float);            
            
}}}
