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

/**
 * @brief See NR 3rd pg. 115.
 *
 */
int Base_interp::locate(double const x) {
    
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
int Base_interp::hunt(double const x) {
    
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
double Poly_interp::rawinterp(int const jl, double const x) {
    int i, m, ns = 0;
    double y, den, dif, dift, ho, hp, w;
    std::vector<double>::pointer xa = &_x[jl];
    std::vector<double>::pointer ya = &_y[jl];
    std::vector<double> c(_m), d(_m);
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


}}}
