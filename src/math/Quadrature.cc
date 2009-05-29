// -*- LSST-C++ -*-
/**
 * @file    Quadrature.cc
 * @ingroup afw
 * @brief   Compute 1- and 2-D integrals with Romberg integration.
 * @author  Steve Bickerton
 * @date    May 25, 2009
 */
#include <vector>
#include <cmath>
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"

#include "lsst/afw/math/Quadrature.h"

namespace pexExceptions = lsst::pex::exceptions;
namespace pexLogging = lsst::pex::logging;

namespace lsst { namespace afw { namespace math {

/* An anonymous namespace for static (local) components.
 *
 * Almost all of the classes/functions in this anonymous namespace
 * are adapted from Numerical Recipes 3'rd Ed.
 */
namespace {

    
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
    class Base_interp {
    public:
        
        Base_interp( std::vector<double> &x, std::vector<double> &y, int m)
            : _x(x), _y(y), _m(m) {
            _jsav = 0;
            _n = x.size();
            _cor = 0;
            _dj = min<int>(1, (int) pow((double) _n, 0.25));
        }
        
        double interp(double x) {
            int const jlo = _cor ? hunt(x) : locate(x);
            return rawinterp(jlo, x);
        }
        
        int locate(const double x);
        int hunt(const double x);
        
        double virtual rawinterp(int jlo, double x) = 0;

    protected:
        std::vector<double> &_x;
        std::vector<double> &_y;
        int _n, _m;
    private:
        int _jsav, _cor, _dj;
    };


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
            if ( (x >= _x[jm]) == ascnd) {
                jl = jm;
            } else {
                ju = jm;
            }
        }
        _cor = abs(jl - _jsav) > _dj ? 0 : 1;
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
        _cor = abs(jl - _jsav) > _dj ? 0 : 1;
        _jsav = jl;
        return max<int>(0, min<int>(_n - _m, jl - ((_m - 2) >> 1)) );
    }


    // =============================================================
    /**
     * @class Poly_interp
     *
     * @brief A Polynomial interpolation object.
     * @note Adapted from NR 3rd Ed. Chap. 3
     */
    class Poly_interp : public Base_interp {
    public:
        Poly_interp(std::vector<double> &xv, std::vector<double>  &yv, int m) :
            Base_interp(xv, yv, m), _dy(0.0) {}
        double rawinterp(int jl, double x);
        double getDy() { return _dy; }
    private:
        double _dy;
        using Base_interp::_x;
        using Base_interp::_y;
        using Base_interp::_n;
        using Base_interp::_m;
    };


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


    // =============================================================
    /**
     * @class Quadrature
     *
     * @brief An integration base class
     * @note Adapted from NR 3rd Ed. Chap. 4
     */
    class Quadrature {
    public:
        virtual double next() = 0;
    protected:
        int _n;
    private:
    };

    
    // =============================================================
    /**
     * @class Trapzd
     * @brief Perform trapezoid-rule integration.
     * @note Adapted from NR 3d Ed. Chap. 4 pg 163
     */
    template<typename FunctionT>
    class Trapzd : public Quadrature {
    public:
        Trapzd() {};
        Trapzd(FunctionT &func, double const a, double const b) :
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
        FunctionT &_func;
        double _a, _b, _s;
        using Quadrature::_n;
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
     * @author S. Bickerton
     */
    class FunctionWrapper {
    public:
        FunctionWrapper(lsst::afw::math::IntegrandBase &func, double x1, double x2)
            : _func(func), _x1(x1), _x2(x2)
            {}
        double operator() (double y) {
            _func.setY(y);
            return romberg(_func, _x1, _x2);
        }
    private:
        IntegrandBase &_func;
        double _x1, _x2;
    };

    
}

    
/**
 * @brief The 1D Romberg integrator
 *
 * @note Adapted from NR 3rd Ed. Chapter 4, pg 166
 *
 * @todo Throw a proper exception when JMAX is exceeded.
 */
template<typename FunctionT>            
double romberg(FunctionT &func, double a, double b, double const eps) {

    static int call_count = 0;
    static int fail_count = 0;
    
    call_count++;
    
    int const JMAX = 20, JMAXP = JMAX + 1, K = 5;
        
    std::vector<double> s(JMAX), h(JMAXP);
    Poly_interp polint(h, s, K);
    h[0] = 1.0;
    Trapzd<FunctionT> t(func, a, b);
    double ss_bail = 0;
    for (int j = 1; j <= JMAX; ++j) {
        s[j - 1] = t.next();
        if (j >= K) {
            double ss = ss_bail = polint.rawinterp(j - K, 0.0);
            if ( (std::fabs(polint.getDy()) < std::fabs(eps*ss)) ||
                 (std::fabs(polint.getDy()) < std::fabs(epsilon_f) ) ) {
                return ss;
            }
        }
        h[j] = 0.25*h[j - 1];
    }
    fail_count++;
    
    // ==== THROW A PROPER EXCEPTION HERE ====
    std::cout << "Failed to converge in " << JMAX << " iterations. fail/call="
              << fail_count <<"/" << call_count << "\n" ;
    return ss_bail;
}
            

/**
 * @brief The 2D Romberg integrator
 * @note Adapted from RHL's SDSS code.
 */
double romberg2D(IntegrandBase &func, double x1, double x2, double y1, double y2, double const eps) {
    FunctionWrapper fwrap(func, x1, x2);
    return romberg<FunctionWrapper>(fwrap, y1, y2, eps);
}
            
}}}


//template double lsst::afw::math::romberg<double (double)>(double &func(double), double a, double b, double const eps);
//template class lsst::afw::math::IntegrandBase<double (double)>;

