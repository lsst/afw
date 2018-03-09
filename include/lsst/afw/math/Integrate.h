// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#if !defined(LSST_AFW_MATH_INTEGRATE_H)
#define LSST_AFW_MATH_INTEGRATE_H 1
/*
 * Compute 1d and 2d integral
 */

#include <functional>
#include <vector>
#include <queue>
#include <map>
#include <cmath>
#include <algorithm>
#include <assert.h>
#include <limits>
#include <ostream>
#include <sstream>
#include <complex>
#include <stdexcept>

#include "lsst/pex/exceptions.h"

#include "lsst/afw/math/IntGKPData10.h"

// == The following is Mike Jarvis original comment ==
//
// Basic Usage:
//
// First, define a function object, which should derive from
// std::unary_function<double,double>.  For example, to integrate a
// Gaussian, use something along the lines of this:
//
// class Gauss :
//   public std::unary_function<double,double>
// {
//   public :
//
//     Gauss(double _mu, double _sig) :
//       mu(_mu), sig(_sig), sigsq(_sig*_sig) {}
//
//     double operator()(double x) const
//     {
//       const double SQRTTWOPI = 2.50662827463;
//       return exp(-pow(x-mu,2)/2./sigsq)/SQRTTWOPI/sig;
//     }
//
//   private :
//     double mu,sig,sigsq;
// };
//
// Next, make an IntRegion object with the bounds of the integration region.
// You need to give it the type to use for the bounds and the value of the
// functions (which need to be the same currently - some day I'll allow
// complex functions...).
//
// For example, to integrate something from -1 to 1, use:
//
// integ::IntRegion<double> reg1(-1.,1.);
//
// (The integ:: is because everything here is in the integ namespace to
// help prevent name conflicts with other header files.)
//
// If a value is > 1.e10 or < -1.e10, then these values are taken to be
// infinity, rather than the actual value.
// So to integrate from 0 to infinity:
//
// integ::IntRegion<double> reg2(0.,1.e100);
//
// Or, you can use the variable integ::MOCK_INF which might be clearer.
//
//
// Finally, to perform the integral, the line would be:
//
// double integ1 = int1d(Gauss(0.,1.),reg1,1.e-10,1.e-4);
// double integ2 = int1d(Gauss(0.,2.),reg2,1.e-10,1.e-4);
//
// which should yield 0.68 and 0.5 in our case.
//
// Those last two numbers indicate the precision required.
// 1.e-10 is the required absolute error, and
// 1.e-4 is the required relative error.
//
// If you want, you can omit these and 1.e-15,1.e-6 will be used as the
// default precision (which are generally fine for most purposes).
//
// The absolute error only comes into play for results which are close to
// 0 to prevent requiring an error of 0 for integrals which evaluate to 0
// or very close to it.
//
//
//
// Advanced Usage:
//
// When an integration fails to converge with the usual GKP algorithm,
// it splits the region into 2 (or more) and tries again with each sub-region.
// The default is to just bisect the region (or something similarly smart for
// infinite regions), but if you know of a good place to split the region,
// you can tell it using:
//
// reg.AddSplit(10.)
//
// For example, if you know that you have a singularity somewhere in your
// region, it would help the program a lot to split there, so you
// should add that as a split point.  Zeros can also be good choices.
//
// In addition to the integral being returned from int1d, int2d, or int3d as
// the return value, the value is also stored in the region itself.
// You can access it using:
//
// reg.Area();
//
// There is also an estimate of the error in the value:
//
// reg.Err();
//
// (It is intended to be an overestimate of the actual error,
// but it doesn't always get it completely right.)
//
//
//
// Two- and Three-Dimensional Integrals:
//
// These are slightly more complicated.  The easiest case is when the
// bounds of the integral are a rectangle or 3d box.  In this case,
// you can still use the regular IntRegion.  The only new thing then
// is the definition of the function.  For example, to integrate
// int(3x^2 + xy + y , x=0..1, y=0..1):
//
// struct Integrand :
//   public std::binary_function<double,double,double>
// {
//   double operator()(double x, double y) const
//   { return x*(3.*x + y) + y; }
// };
//
// integ::IntRegion<double> reg3(0.,1.);
// double integ3 = int2d(Integrand(),reg3,reg3);
//
// (Which should give 1.75 as the result.)
//
//
//

namespace lsst {
namespace afw {
namespace math {

double const MOCK_INF = 1.e10;

#ifdef NDEBUG
#define integ_dbg1 \
    if (false) (*_dbgout)
#define integ_dbg2 \
    if (false) (*(reg.getDbgout()))
#define integ_dbg3 \
    if (false) (*(tempreg.getDbgout()))
#else
#define integ_dbg1 \
    if (_dbgout) (*_dbgout)
#define integ_dbg2 \
    if (reg.getDbgout()) (*(reg.getDbgout()))
#define integ_dbg3 \
    if (tempreg.getDbgout()) (*(tempreg.getDbgout()))
#endif

//#define COUNTFEVAL
// If defined, then count the number of function evaluations

namespace details {
template <class T>
inline T norm(T const &x) {
    return x * x;
}
using std::norm;
template <class T>
inline T real(T const &x) {
    return x;
}
using std::real;
#ifdef COUNTFEVAL
int nfeval = 0;
#endif
}

template <class T>
struct IntRegion {
public:
    IntRegion(T const a, T const b, std::ostream *dbgout = 0)
            : _a(a), _b(b), _error(0.0), _area(0), _dbgout(dbgout) {}

    IntRegion(IntRegion const &) = default;
    IntRegion(IntRegion &&) = default;
    IntRegion & operator=(IntRegion const &) = default;
    IntRegion & operator=(IntRegion &&) = default;
    ~IntRegion() = default;

    bool operator<(IntRegion<T> const &r2) const { return _error < r2._error; }
    bool operator>(IntRegion<T> const &r2) const { return _error > r2._error; }

    void SubDivide(std::vector<IntRegion<T> > *children) {
        assert(children->size() == 0);
        if (_splitpoints.size() == 0) {
            Bisect();
        }
        if (_splitpoints.size() > 1) {
            std::sort(_splitpoints.begin(), _splitpoints.end());
        }

#if 0
        if (_a > _splitpoints[0] || _b < _splitpoints.back()) {
            std::cerr << "a, b = " << _a << ', ' << _b << std::endl;
            std::cerr << "_splitpoints = ";
            for (size_t i = 0; i<_splitpoints.size(); i++)  {
                std::cerr << _splitpoints[i] << "  ";
            }
            std::cerr << std::endl;
        }
#endif
        assert(_splitpoints[0] >= _a);
        assert(_splitpoints.back() <= _b);
        children->push_back(IntRegion<T>(_a, _splitpoints[0], _dbgout));
        for (size_t i = 1; i < _splitpoints.size(); i++) {
            children->push_back(IntRegion<T>(_splitpoints[i - 1], _splitpoints[i], _dbgout));
        }
        children->push_back(IntRegion<T>(_splitpoints.back(), _b, _dbgout));
    }

    void Bisect() { _splitpoints.push_back((_a + _b) / 2.0); }
    void AddSplit(const T x) { _splitpoints.push_back(x); }
    size_t NSplit() const { return _splitpoints.size(); }

    T const &Left() const { return _a; }
    T const &Right() const { return _b; }
    T const &Err() const { return _error; }
    T const &Area() const { return _area; }
    void SetArea(T const &a, T const &e) {
        _area = a;
        _error = e;
    }

    std::ostream *getDbgout() { return _dbgout; }

private:
    T _a, _b, _error, _area;
    std::vector<T> _splitpoints;
    std::ostream *_dbgout;
};

double const DEFABSERR = 1.e-15;
double const DEFRELERR = 1.e-6;

namespace details {

template <class T>
inline T Epsilon() {
    return std::numeric_limits<T>::epsilon();
}
template <class T>
inline T MinRep() {
    return std::numeric_limits<T>::min();
}

#ifdef EXTRA_PREC_H
template <>
inline Quad Epsilon<Quad>() {
    return 3.08148791094e-33;
}
template <>
inline Quad MinRep<Quad>() {
    return 2.2250738585072014e-308;
}
#endif

template <class T>
inline T rescale_error(T err, T const &resabs, T const &resasc) {
    if (resasc != 0.0 && err != 0.0) {
        T const scale = (200.0 * err / resasc);
        if (scale < 1.0) {
            err = resasc * scale * sqrt(scale);
        } else {
            err = resasc;
        }
    }
    if (resabs > MinRep<T>() / (50.0 * Epsilon<T>())) {
        T const min_err = 50.0 * Epsilon<T>() * resabs;
        if (min_err > err) {
            err = min_err;
        }
    }
    return err;
}

/**
 * Non-adaptive integration of the function f over the region 'reg'.
 *
 * @note The algorithm computes first a Gaussian quadrature value
 *       then successive Kronrod/Patterson extensions to this result.
 *       The functions terminates when the difference between successive
 *       approximations (rescaled according to rescale_error) is less than
 *       either epsabs or epsrel * I, where I is the latest estimate of the
 *       integral.
 *       The order of the Gauss/Kronron/Patterson scheme is determined
 *       by which file is included above.  Currently schemes starting
 *       with order 1 and order 10 are calculated.  There seems to be
 *       little practical difference in the integration times using
 *       the two schemes, so I haven't bothered to calculate any more.
 */

template <class UF>
inline bool intGKPNA(UF const &func, IntRegion<typename UF::result_type> &reg,
                     typename UF::result_type const epsabs, typename UF::result_type const epsrel,
                     std::map<typename UF::result_type, typename UF::result_type> *fxmap = 0) {
    typedef typename UF::result_type UfResult;
    UfResult const a = reg.Left();
    UfResult const b = reg.Right();

    UfResult const halfLength = 0.5 * (b - a);
    UfResult const absHalfLength = fabs(halfLength);
    UfResult const center = 0.5 * (b + a);
    UfResult const fCenter = func(center);
#ifdef COUNTFEVAL
    nfeval++;
#endif

    assert(gkp_wb<UfResult>(0).size() == gkp_x<UfResult>(0).size() + 1);
    UfResult area1 = gkp_wb<UfResult>(0).back() * fCenter;
    std::vector<UfResult> fv1, fv2;
    fv1.reserve(2 * gkp_x<UfResult>(0).size() + 1);
    fv2.reserve(2 * gkp_x<UfResult>(0).size() + 1);
    for (size_t k = 0; k < gkp_x<UfResult>(0).size(); k++) {
        UfResult const abscissa = halfLength * gkp_x<UfResult>(0)[k];
        UfResult const fval1 = func(center - abscissa);
        UfResult const fval2 = func(center + abscissa);
        area1 += gkp_wb<UfResult>(0)[k] * (fval1 + fval2);
        fv1.push_back(fval1);
        fv2.push_back(fval2);
        if (fxmap) {
            (*fxmap)[center - abscissa] = fval1;
            (*fxmap)[center + abscissa] = fval2;
        }
    }
#ifdef COUNTFEVAL
    nfeval += gkp_x<UfResult>(0).size() * 2;
#endif

    integ_dbg2 << "level 0 rule: area = " << area1 << std::endl;

    UfResult err = 0;
    bool calcabsasc = true;
    UfResult resabs = 0.0, resasc = 0.0;
    for (int level = 1; level < NGKPLEVELS; level++) {
        assert(gkp_wa<UfResult>(level).size() == fv1.size());
        assert(gkp_wa<UfResult>(level).size() == fv2.size());
        assert(gkp_wb<UfResult>(level).size() == gkp_x<UfResult>(level).size() + 1);
        UfResult area2 = gkp_wb<UfResult>(level).back() * fCenter;
        // resabs = approximation to integral of abs(f)
        if (calcabsasc) {
            resabs = fabs(area2);
        }
        for (size_t k = 0; k < fv1.size(); k++) {
            area2 += gkp_wa<UfResult>(level)[k] * (fv1[k] + fv2[k]);
            if (calcabsasc) {
                resabs += gkp_wa<UfResult>(level)[k] * (fabs(fv1[k]) + fabs(fv2[k]));
            }
        }
        for (size_t k = 0; k < gkp_x<UfResult>(level).size(); k++) {
            UfResult const abscissa = halfLength * gkp_x<UfResult>(level)[k];
            UfResult const fval1 = func(center - abscissa);
            UfResult const fval2 = func(center + abscissa);
            UfResult const fval = fval1 + fval2;
            area2 += gkp_wb<UfResult>(level)[k] * fval;
            if (calcabsasc) {
                resabs += gkp_wb<UfResult>(level)[k] * (fabs(fval1) + fabs(fval2));
            }
            fv1.push_back(fval1);
            fv2.push_back(fval2);
            if (fxmap) {
                (*fxmap)[center - abscissa] = fval1;
                (*fxmap)[center + abscissa] = fval2;
            }
        }
#ifdef COUNTFEVAL
        nfeval += gkp_x<UfResult>(level).size() * 2;
#endif
        if (calcabsasc) {
            UfResult const mean = area1 * UfResult(0.5);
            // resasc = approximation to the integral of abs(f-mean)
            resasc = gkp_wb<UfResult>(level).back() * fabs(fCenter - mean);
            for (size_t k = 0; k < gkp_wa<UfResult>(level).size(); k++) {
                resasc += gkp_wa<UfResult>(level)[k] * (fabs(fv1[k] - mean) + fabs(fv2[k] - mean));
            }
            for (size_t k = 0; k < gkp_x<UfResult>(level).size(); k++) {
                resasc += gkp_wb<UfResult>(level)[k] * (fabs(fv1[k] - mean) + fabs(fv2[k] - mean));
            }
            resasc *= absHalfLength;
            resabs *= absHalfLength;
        }
        area2 *= halfLength;
        err = rescale_error(fabs(area2 - area1), resabs, resasc);
        if (err < resasc) {
            calcabsasc = false;
        }

        integ_dbg2 << "at level " << level << " area2 = " << area2;
        integ_dbg2 << " +- " << err << std::endl;

        //   test for convergence.
        if (err < epsabs || err < epsrel * fabs(area2)) {
            reg.SetArea(area2, err);
            return true;
        }
        area1 = area2;
    }

    // failed to converge
    reg.SetArea(area1, err);

    integ_dbg2 << "Failed to reach tolerance with highest-order GKP rule";

    return false;
}

/**
 * An adaptive integration algorithm which computes the integral of f over the region reg.
 *
 * @note First the non-adaptive GKP algorithm is tried.
 *       If that is not accurate enough (according to the absolute and
 *       relative accuracies, epsabs and epsrel),
 *       the region is split in half, and each new region is integrated.
 *       The routine continues by successively splitting the subregion
 *       which gave the largest absolute error until the integral converges.
 *
 *       The area and estimated error are returned as reg.Area() and reg.Err()
 *       If desired, *retx and *retf return std::vectors of x,f(x) respectively
 *       They only include the evaluations in the non-adaptive pass, so they
 *       do not give an accurate estimate of the number of function evaluations.
 */

template <class UF>
inline void intGKP(UF const &func, IntRegion<typename UF::result_type> &reg,
                   typename UF::result_type const epsabs, typename UF::result_type const epsrel,
                   std::map<typename UF::result_type, typename UF::result_type> *fxmap = 0) {
    typedef typename UF::result_type UfResult;
    integ_dbg2 << "Start intGKP\n";

    assert(epsabs >= 0.0);
    assert(epsrel > 0.0);

    // perform the first integration
    bool done = intGKPNA(func, reg, epsabs, epsrel, fxmap);
    if (done) return;

    integ_dbg2 << "In adaptive GKP, failed first pass... subdividing\n";
    integ_dbg2 << "Intial range = " << reg.Left() << ".." << reg.Right() << std::endl;

    int roundoffType1 = 0, errorType = 0;
    UfResult roundoffType2 = 0;
    size_t iteration = 1;

    std::priority_queue<IntRegion<UfResult>, std::vector<IntRegion<UfResult> > > allregions;
    allregions.push(reg);
    UfResult finalarea = reg.Area();
    UfResult finalerr = reg.Err();
    UfResult tolerance = std::max(epsabs, epsrel * fabs(finalarea));
    assert(finalerr > tolerance);

    while (!errorType && finalerr > tolerance) {
        // Bisect the subinterval with the largest error estimate
        integ_dbg2 << "Current answer = " << finalarea << " +- " << finalerr;
        integ_dbg2 << "  (tol = " << tolerance << ")\n";
        IntRegion<UfResult> parent = allregions.top();
        allregions.pop();
        integ_dbg2 << "Subdividing largest error region ";
        integ_dbg2 << parent.Left() << ".." << parent.Right() << std::endl;
        integ_dbg2 << "parent area = " << parent.Area();
        integ_dbg2 << " +- " << parent.Err() << std::endl;
        std::vector<IntRegion<UfResult> > children;
        parent.SubDivide(&children);
        // For "GKP", there are only two, but for GKPOSC, there is one
        // for each oscillation in region

        // Try to do at least 3x better with the children
        UfResult factor = 3 * children.size() * finalerr / tolerance;
        UfResult newepsabs = fabs(parent.Err() / factor);
        UfResult newepsrel = newepsabs / fabs(parent.Area());
        integ_dbg2 << "New epsabs, rel = " << newepsabs << ", " << newepsrel;
        integ_dbg2 << "  (" << children.size() << " children)\n";

        UfResult newarea = UfResult(0.0);
        UfResult newerror = 0.0;
        for (size_t i = 0; i < children.size(); i++) {
            IntRegion<UfResult> &child = children[i];
            integ_dbg2 << "Integrating child " << child.Left();
            integ_dbg2 << ".." << child.Right() << std::endl;
            bool hasConverged;
            hasConverged = intGKPNA(func, child, newepsabs, newepsrel);
            integ_dbg2 << "child (" << i + 1 << '/' << children.size() << ") ";
            if (hasConverged) {
                integ_dbg2 << " converged.";
            } else {
                integ_dbg2 << " failed.";
            }
            integ_dbg2 << "  Area = " << child.Area() << " +- " << child.Err() << std::endl;

            newarea += child.Area();
            newerror += child.Err();
        }
        integ_dbg2 << "Compare: newerr = " << newerror;
        integ_dbg2 << " to parent err = " << parent.Err() << std::endl;

        finalerr += (newerror - parent.Err());
        finalarea += newarea - parent.Area();

        UfResult delta = parent.Area() - newarea;
        if (newerror <= parent.Err() && fabs(delta) <= parent.Err() && newerror >= 0.99 * parent.Err()) {
            integ_dbg2 << "roundoff type 1: delta/newarea = ";
            integ_dbg2 << fabs(delta) / fabs(newarea);
            integ_dbg2 << ", newerror/error = " << newerror / parent.Err() << std::endl;
            roundoffType1++;
        }
        if (iteration >= 10 && newerror > parent.Err() && fabs(delta) <= newerror - parent.Err()) {
            integ_dbg2 << "roundoff type 2: newerror/error = ";
            integ_dbg2 << newerror / parent.Err() << std::endl;
            roundoffType2 += std::min(newerror / parent.Err() - 1.0, UfResult(1.0));
        }

        tolerance = std::max(epsabs, epsrel * fabs(finalarea));
        if (finalerr > tolerance) {
            if (roundoffType1 >= 200) {
                errorType = 1;  // round off error
                integ_dbg2 << "GKP: Round off error 1\n";
            }
            if (roundoffType2 >= 200.0) {
                errorType = 2;  // round off error
                integ_dbg2 << "GKP: Round off error 2\n";
            }
            if (fabs((parent.Right() - parent.Left()) / (reg.Right() - reg.Left())) < Epsilon<double>()) {
                errorType = 3;  // found singularity
                integ_dbg2 << "GKP: Probable singularity\n";
            }
        }
        for (size_t i = 0; i < children.size(); i++) {
            allregions.push(children[i]);
        }
        iteration++;
    }

    // Recalculate finalarea in case there are any slight rounding errors
    finalarea = 0.0;
    finalerr = 0.0;
    while (!allregions.empty()) {
        IntRegion<UfResult> const &r = allregions.top();
        finalarea += r.Area();
        finalerr += r.Err();
        allregions.pop();
    }
    reg.SetArea(finalarea, finalerr);

    if (errorType == 1) {
        std::ostringstream s;
        s << "Type 1 roundoff's = " << roundoffType1;
        s << ", Type 2 = " << roundoffType2 << std::endl;
        s << "Roundoff error 1 prevents tolerance from being achieved ";
        s << "in intGKP\n";
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, s.str());
    } else if (errorType == 2) {
        std::ostringstream s;
        s << "Type 1 roundoff's = " << roundoffType1;
        s << ", Type 2 = " << roundoffType2 << std::endl;
        s << "Roundoff error 2 prevents tolerance from being achieved ";
        s << "in intGKP\n";
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, s.str());
    } else if (errorType == 3) {
        std::ostringstream s;
        s << "Bad integrand behavior found in the integration interval ";
        s << "in intGKP\n";
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, s.str());
    }
}

/**
 * Auxiliary struct 1
 *
 */
template <class UF>
struct AuxFunc1 :  // f(1/x-1) for int(a..infinity)
                   public std::unary_function<typename UF::argument_type, typename UF::result_type> {
public:
    AuxFunc1(UF const &f) : _f(f) {}
    typename UF::result_type operator()(typename UF::argument_type x) const {
        return _f(1.0 / x - 1.0) / (x * x);
    }

private:
    UF const &_f;
};

/**
 * Auxiliary function 1
 *
 */
template <class UF>
AuxFunc1<UF> inline Aux1(UF uf) {
    return AuxFunc1<UF>(uf);
}

template <class UF>
struct AuxFunc2 :  // f(1/x+1) for int(-infinity..b)
                   public std::unary_function<typename UF::argument_type, typename UF::result_type> {
public:
    AuxFunc2(UF const &f) : _f(f) {}
    typename UF::result_type operator()(typename UF::argument_type x) const {
        return _f(1.0 / x + 1.0) / (x * x);
    }

private:
    UF const &_f;
};

/**
 * Auxiliary function 2
 *
 */
template <class UF>
AuxFunc2<UF> inline Aux2(UF uf) {
    return AuxFunc2<UF>(uf);
}

/**
 * Helpers for constant regions for int2d, int3d:
 *
 */
template <class T>
struct ConstantReg1 : public std::unary_function<T, IntRegion<T> > {
    ConstantReg1(T a, T b) : ir(a, b) {}
    ConstantReg1(IntRegion<T> const &r) : ir(r) {}
    IntRegion<T> operator()(T) const { return ir; }
    IntRegion<T> ir;
};

template <class T>
struct ConstantReg2 : public std::binary_function<T, T, IntRegion<T> > {
    ConstantReg2(T a, T b) : ir(a, b) {}
    ConstantReg2(IntRegion<T> const &r) : ir(r) {}
    IntRegion<T> operator()(T x, T y) const { return ir; }
    IntRegion<T> ir;
};

// pulled from MoreFunctional.h.  Needed in class Int2DAuxType and Int3DAuxType
template <class BF>
class binder2_1 : public std::unary_function<typename BF::second_argument_type, typename BF::result_type> {
public:
    binder2_1(BF const &oper, typename BF::first_argument_type val) : _oper(oper), _value(val) {}
    typename BF::result_type operator()(const typename BF::second_argument_type &x) const {
        return _oper(_value, x);
    }

protected:
    BF _oper;
    typename BF::first_argument_type _value;
};

template <class BF, class Tp>
inline binder2_1<BF> bind21(BF const &oper, Tp const &x) {
    typedef typename BF::first_argument_type Arg;
    return binder2_1<BF>(oper, static_cast<Arg>(x));
}

template <class BF, class YREG>
class Int2DAuxType : public std::unary_function<typename BF::first_argument_type, typename BF::result_type> {
public:
    Int2DAuxType(BF const &func, YREG const &yreg, typename BF::result_type const &abserr,
                 typename BF::result_type const &relerr)
            : _func(func), _yreg(yreg), _abserr(abserr), _relerr(relerr) {}

    typename BF::result_type operator()(typename BF::first_argument_type x) const {
        typename YREG::result_type tempreg = _yreg(x);
        typename BF::result_type result = int1d(bind21(_func, x), tempreg, _abserr, _relerr);
        integ_dbg3 << "Evaluated int2dAux at x = " << x;
        integ_dbg3 << ": f = " << result << " +- " << tempreg.Err() << std::endl;
        return result;
    }

private:
    BF const &_func;
    YREG const &_yreg;
    typename BF::result_type _abserr, _relerr;
};

// pulled from MoreFunctional.h.  Needed in class Int3DAuxtype
template <class TF>
class binder3_1 : public std::binary_function<typename TF::secondof3_argument_type,
                                              typename TF::thirdof3_argument_type, typename TF::result_type> {
public:
    binder3_1(TF const &oper, typename TF::firstof3_argument_type val) : _oper(oper), _value(val) {}
    typename TF::result_type operator()(typename TF::secondof3_argument_type const &x1,
                                        typename TF::thirdof3_argument_type const &x2) const {
        return _oper(_value, x1, x2);
    }

protected:
    TF _oper;
    typename TF::firstof3_argument_type _value;
};

template <class TF, class Tp>
inline binder3_1<TF> bind31(TF const &oper, Tp const &x) {
    typedef typename TF::firstof3_argument_type Arg;
    return binder3_1<TF>(oper, static_cast<Arg>(x));
}

template <class TF, class YREG, class ZREG>
class Int3DAuxType
        : public std::unary_function<typename TF::firstof3_argument_type, typename TF::result_type> {
public:
    Int3DAuxType(TF const &func, YREG const &yreg, ZREG const &zreg, const typename TF::result_type &abserr,
                 const typename TF::result_type &relerr)
            : _func(func), _yreg(yreg), _zreg(zreg), _abserr(abserr), _relerr(relerr) {}

    typename TF::result_type operator()(typename TF::firstof3_argument_type x) const {
        typename YREG::result_type tempreg = _yreg(x);
        typename TF::result_type result =
                int2d(bind31(_func, x), tempreg, bind21(_zreg, x), _abserr, _relerr);
        integ_dbg3 << "Evaluated int3dAux at x = " << x;
        integ_dbg3 << ": f = " << result << " +- " << tempreg.Err() << std::endl;
        return result;
    }

private:
    TF const &_func;
    YREG const &_yreg;
    ZREG const &_zreg;
    typename TF::result_type _abserr, _relerr;
};

}  // end namespace details

/**
 * Front end for the 1d integrator
 */
template <class UF>
inline typename UF::result_type int1d(UF const &func, IntRegion<typename UF::result_type> &reg,
                                      typename UF::result_type const &abserr = DEFABSERR,
                                      typename UF::result_type const &relerr = DEFRELERR) {
    typedef typename UF::result_type UfResult;
    using namespace details;

    integ_dbg2 << "start int1d: " << reg.Left() << ".." << reg.Right() << std::endl;

    if ((reg.Left() <= -MOCK_INF && reg.Right() > 0) || (reg.Right() >= MOCK_INF && reg.Left() < 0)) {
        reg.AddSplit(0);
    }

    if (reg.NSplit() > 0) {
        std::vector<IntRegion<UfResult> > children;
        reg.SubDivide(&children);
        integ_dbg2 << "Subdivided into " << children.size() << " children\n";
        UfResult answer = UfResult();
        UfResult err = 0;
        for (size_t i = 0; i < children.size(); i++) {
            IntRegion<UfResult> &child = children[i];
            integ_dbg2 << "i = " << i;
            integ_dbg2 << ": bounds = " << child.Left() << ", " << child.Right() << std::endl;
            answer += int1d(func, child, abserr, relerr);
            err += child.Err();
            integ_dbg2 << "subint = " << child.Area() << " +- " << child.Err() << std::endl;
        }
        reg.SetArea(answer, err);
        return answer;

    } else {
        if (reg.Left() <= -MOCK_INF) {
            integ_dbg2 << "left = -infinity, right = " << reg.Right() << std::endl;
            assert(reg.Right() <= 0.0);
            IntRegion<UfResult> modreg(1.0 / (reg.Right() - 1.0), 0.0, reg.getDbgout());
            intGKP(Aux2<UF>(func), modreg, abserr, relerr);
            reg.SetArea(modreg.Area(), modreg.Err());
        } else if (reg.Right() >= MOCK_INF) {
            integ_dbg2 << "left = " << reg.Left() << ", right = infinity\n";
            assert(reg.Left() >= 0.0);
            IntRegion<UfResult> modreg(0.0, 1.0 / (reg.Left() + 1.0), reg.getDbgout());
            intGKP(Aux1<UF>(func), modreg, abserr, relerr);
            reg.SetArea(modreg.Area(), modreg.Err());
        } else {
            integ_dbg2 << "left = " << reg.Left();
            integ_dbg2 << ", right = " << reg.Right() << std::endl;
            intGKP(func, reg, abserr, relerr);
        }
        integ_dbg2 << "done int1d  answer = " << reg.Area();
        integ_dbg2 << " +- " << reg.Err() << std::endl;
        return reg.Area();
    }
}

/**
 * Front end for the 2d integrator
 */
template <class BF, class YREG>
inline typename BF::result_type int2d(BF const &func, IntRegion<typename BF::result_type> &reg,
                                      YREG const &yreg, typename BF::result_type const &abserr = DEFABSERR,
                                      typename BF::result_type const &relerr = DEFRELERR) {
    using namespace details;
    integ_dbg2 << "Starting int2d: range = ";
    integ_dbg2 << reg.Left() << ".." << reg.Right() << std::endl;
    Int2DAuxType<BF, YREG> faux(func, yreg, abserr * 1.0e-3, relerr * 1.0e-3);
    typename BF::result_type answer = int1d(faux, reg, abserr, relerr);
    integ_dbg2 << "done int2d  answer = " << answer << " +- " << reg.Err() << std::endl;
    return answer;
}

/**
 * Front end for the 3d integrator
 */
template <class TF, class YREG, class ZREG>
inline typename TF::result_type int3d(TF const &func, IntRegion<typename TF::result_type> &reg,
                                      YREG const &yreg, ZREG const &zreg,
                                      typename TF::result_type const &abserr = DEFABSERR,
                                      typename TF::result_type const &relerr = DEFRELERR) {
    using namespace details;
    integ_dbg2 << "Starting int3d: range = ";
    integ_dbg2 << reg.Left() << ".." << reg.Right() << std::endl;
    Int3DAuxType<TF, YREG, ZREG> faux(func, yreg, zreg, abserr * 1.e-3, relerr * 1.e-3);
    typename TF::result_type answer = int1d(faux, reg, abserr, relerr);
    integ_dbg2 << "done int3d  answer = " << answer << " +- " << reg.Err() << std::endl;
    return answer;
}

/**
 * Front end for the 2d integrator
 */
template <class BF>
inline typename BF::result_type int2d(BF const &func, IntRegion<typename BF::result_type> &reg,
                                      IntRegion<typename BF::result_type> &yreg,
                                      typename BF::result_type const &abserr = DEFABSERR,
                                      typename BF::result_type const &relerr = DEFRELERR) {
    using namespace details;
    return int2d(func, reg, ConstantReg1<typename BF::result_type>(yreg), abserr, relerr);
}

/**
 * Front end for the 3d integrator
 */
template <class TF>
inline typename TF::result_type int3d(TF const &func, IntRegion<typename TF::result_type> &reg,
                                      IntRegion<typename TF::result_type> &yreg,
                                      IntRegion<typename TF::result_type> &zreg,
                                      typename TF::result_type const &abserr = DEFABSERR,
                                      typename TF::result_type const &relerr = DEFRELERR) {
    using namespace details;
    return int3d(func, reg, ConstantReg1<typename TF::result_type>(yreg),
                 ConstantReg2<typename TF::result_type>(zreg), abserr, relerr);
}

// =============================================================
/**
 * The 1D integrator
 *
 * @note This simply wraps the int1d function above and handles the
 *       instantiation of the intRegion.
 *
 */
template <typename UnaryFunctionT>
typename UnaryFunctionT::result_type integrate(UnaryFunctionT func,
                                               typename UnaryFunctionT::argument_type const a,
                                               typename UnaryFunctionT::argument_type const b,
                                               double eps = 1.0e-6) {
    typedef typename UnaryFunctionT::argument_type Arg;
    IntRegion<Arg> region(a, b);

    return int1d(func, region, DEFABSERR, eps);
}

namespace details {

/**
 * Wrap an integrand in a call to a 1D integrator: romberg()
 *
 * When romberg2D() is called, it wraps the integrand it was given
 * in a FunctionWrapper functor.  This wrapper calls romberg() on the integrand
 * to get a 1D (along the x-coord, for constant y) result .
 * romberg2D() then calls romberg() with the FunctionWrapper functor as an
 * integrand.
 */
template <typename BinaryFunctionT>
class FunctionWrapper : public std::unary_function<typename BinaryFunctionT::second_argument_type,
                                                   typename BinaryFunctionT::result_type> {
public:
    FunctionWrapper(BinaryFunctionT func, typename BinaryFunctionT::first_argument_type const x1,
                    typename BinaryFunctionT::first_argument_type const x2, double const eps = 1.0e-6)
            : _func(func), _x1(x1), _x2(x2), _eps(eps) {}
    typename BinaryFunctionT::result_type operator()(
            typename BinaryFunctionT::second_argument_type const y) const {
        return integrate(std::bind2nd(_func, y), _x1, _x2, _eps);
    }

private:
    BinaryFunctionT _func;
    typename BinaryFunctionT::first_argument_type _x1, _x2;
    double _eps;
};

}  // end of namespace afw::math::details

// =============================================================
/**
 * The 2D integrator
 *
 * @note Adapted from RHL's SDSS code
 */

template <typename BinaryFunctionT>
typename BinaryFunctionT::result_type integrate2d(BinaryFunctionT func,
                                                  typename BinaryFunctionT::first_argument_type const x1,
                                                  typename BinaryFunctionT::first_argument_type const x2,
                                                  typename BinaryFunctionT::second_argument_type const y1,
                                                  typename BinaryFunctionT::second_argument_type const y2,
                                                  double eps = 1.0e-6) {
    using namespace details;
    // note the more stringent eps requirement to ensure the requested limit
    // can be reached.
    FunctionWrapper<BinaryFunctionT> fwrap(func, x1, x2, eps);
    return integrate(fwrap, y1, y2, eps);
}
}
}
}  // end namespaces lsst/afw/math

#endif
