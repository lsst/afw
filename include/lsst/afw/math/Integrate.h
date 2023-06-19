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
 * Compute 1d
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <map>
#include <ostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "lsst/pex/exceptions.h"

#include "lsst/afw/math/IntGKPData10.h"

// == The following is based on Mike Jarvis original comment ==
//
// Basic Usage:
//
// First, define a function object.
// For example, to integrate a
// Gaussian, use something along the lines of this:
//
// class Gauss {
//         public :
//         Gauss(double _mu, double _sig) : mu(_mu), sig(_sig) {}
//         double operator()(double x) const {
//             constexpr double inv_sqrt_2pi = 0.3989422804014327;
//             double a = (x - mu) / sig;
//             return inv_sqrt_2pi / sig * std::exp(-0.5 * a * a);
//         }
//         private :
//             double mu,sig;
// };
//
// Next, make an IntRegion object with the bounds of the integration region.
// You need to give it the type to use for the bounds and the value of the
// functions (which need to be the same currently - some day I'll allow
// complex functions...).
//
// For example, to integrate something from -1 to 1, use:
//
// lsst::afw::math::IntRegion<double> reg1(-1.,1.);
// If a value is > 1.e10 or < -1.e10, then these values are taken to be
// infinity, rather than the actual value.
// So to integrate from 0 to infinity:
//
// lsst::afw::math::IntRegion<double> reg2(0.,20);
//
// Or, you can use the variable lsst::afw::math:MOCK_INF which might be clearer.
// The integral might diverge depending on the limits chosen
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
inline T norm(const T &x) {
    return x * x;
}
using std::norm;
template <class T>
inline T real(const T &x) {
    return x;
}
using std::real;
#ifdef COUNTFEVAL
int nfeval = 0;
#endif
}  // namespace details

template <class T>
struct IntRegion final {
public:
    IntRegion(T const a, T const b, std::ostream *dbgout = nullptr)
            : _a(a), _b(b), _error(0.0), _area(0), _dbgout(dbgout) {}

    IntRegion(IntRegion const &) = default;
    IntRegion(IntRegion &&) = default;
    IntRegion &operator=(IntRegion const &) = default;
    IntRegion &operator=(IntRegion &&) = default;
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
    void SetArea(const T &a, const T &e) {
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
template <typename UnaryFunctionT, typename Arg>
inline bool intGKPNA(UnaryFunctionT func, IntRegion<Arg> &reg,
		     Arg const epsabs, Arg const epsrel,
		     std::map<Arg, Arg> *fxmap = nullptr) {
    Arg const a = reg.Left();
    Arg const b = reg.Right();

    Arg const halfLength = 0.5 * (b - a);
    Arg const absHalfLength = fabs(halfLength);
    Arg const center = 0.5 * (b + a);
    Arg const fCenter = func(center);
#ifdef COUNTFEVAL
    nfeval++;
#endif

    assert(gkp_wb<Arg>(0).size() == gkp_x<Arg>(0).size() + 1);
    Arg area1 = gkp_wb<Arg>(0).back() * fCenter;
    std::vector<Arg> fv1, fv2;
    fv1.reserve(2 * gkp_x<Arg>(0).size() + 1);
    fv2.reserve(2 * gkp_x<Arg>(0).size() + 1);
    for (size_t k = 0; k < gkp_x<Arg>(0).size(); k++) {
        Arg const abscissa = halfLength * gkp_x<Arg>(0)[k];
        Arg const fval1 = func(center - abscissa);
        Arg const fval2 = func(center + abscissa);
        area1 += gkp_wb<Arg>(0)[k] * (fval1 + fval2);
        fv1.push_back(fval1);
        fv2.push_back(fval2);
        if (fxmap) {
            (*fxmap)[center - abscissa] = fval1;
            (*fxmap)[center + abscissa] = fval2;
        }
    }
#ifdef COUNTFEVAL
    nfeval += gkp_x<Arg>(0).size() * 2;
#endif

    integ_dbg2 << "level 0 rule: area = " << area1 << std::endl;

    Arg err = 0;
    bool calcabsasc = true;
    Arg resabs = 0.0, resasc = 0.0;
    for (int level = 1; level < NGKPLEVELS; level++) {
        assert(gkp_wa<Arg>(level).size() == fv1.size());
        assert(gkp_wa<Arg>(level).size() == fv2.size());
        assert(gkp_wb<Arg>(level).size() == gkp_x<Arg>(level).size() + 1);
        Arg area2 = gkp_wb<Arg>(level).back() * fCenter;
        // resabs = approximation to integral of abs(f)
        if (calcabsasc) {
            resabs = fabs(area2);
        }
        for (size_t k = 0; k < fv1.size(); k++) {
            area2 += gkp_wa<Arg>(level)[k] * (fv1[k] + fv2[k]);
            if (calcabsasc) {
                resabs += gkp_wa<Arg>(level)[k] * (fabs(fv1[k]) + fabs(fv2[k]));
            }
        }
        for (size_t k = 0; k < gkp_x<Arg>(level).size(); k++) {
            Arg const abscissa = halfLength * gkp_x<Arg>(level)[k];
            Arg const fval1 = func(center - abscissa);
            Arg const fval2 = func(center + abscissa);
            Arg const fval = fval1 + fval2;
            area2 += gkp_wb<Arg>(level)[k] * fval;
            if (calcabsasc) {
                resabs += gkp_wb<Arg>(level)[k] * (fabs(fval1) + fabs(fval2));
            }
            fv1.push_back(fval1);
            fv2.push_back(fval2);
            if (fxmap) {
                (*fxmap)[center - abscissa] = fval1;
                (*fxmap)[center + abscissa] = fval2;
            }
        }
#ifdef COUNTFEVAL
        nfeval += gkp_x<Arg>(level).size() * 2;
#endif
        if (calcabsasc) {
            Arg const mean = area1 * Arg(0.5);
            // resasc = approximation to the integral of abs(f-mean)
            resasc = gkp_wb<Arg>(level).back() * fabs(fCenter - mean);
            for (size_t k = 0; k < gkp_wa<Arg>(level).size(); k++) {
                resasc += gkp_wa<Arg>(level)[k] * (fabs(fv1[k] - mean) + fabs(fv2[k] - mean));
            }
            for (size_t k = 0; k < gkp_x<Arg>(level).size(); k++) {
                resasc += gkp_wb<Arg>(level)[k] * (fabs(fv1[k] - mean) + fabs(fv2[k] - mean));
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
template <typename UnaryFunctionT, typename Arg>
inline void intGKP(UnaryFunctionT func, IntRegion<Arg> &reg,
		   Arg const epsabs, Arg const epsrel,
		   std::map<Arg, Arg> *fxmap = nullptr) {
    integ_dbg2 << "Start intGKP\n";

    assert(epsabs >= 0.0);
    assert(epsrel > 0.0);

    // perform the first integration
    bool done = intGKPNA(func, reg, epsabs, epsrel, fxmap);
    if (done) return;

    integ_dbg2 << "In adaptive GKP, failed first pass... subdividing\n";
    integ_dbg2 << "Intial range = " << reg.Left() << ".." << reg.Right() << std::endl;

    int roundoffType1 = 0, errorType = 0;
    Arg roundoffType2 = 0;
    size_t iteration = 1;

    std::priority_queue<IntRegion<Arg>, std::vector<IntRegion<Arg> > > allregions;
    allregions.push(reg);
    Arg finalarea = reg.Area();
    Arg finalerr = reg.Err();
    Arg tolerance = std::max(epsabs, epsrel * fabs(finalarea));
    assert(finalerr > tolerance);

    while (!errorType && finalerr > tolerance) {
        // Bisect the subinterval with the largest error estimate
        integ_dbg2 << "Current answer = " << finalarea << " +- " << finalerr;
        integ_dbg2 << "  (tol = " << tolerance << ")\n";
        IntRegion<Arg> parent = allregions.top();
        allregions.pop();
        integ_dbg2 << "Subdividing largest error region ";
        integ_dbg2 << parent.Left() << ".." << parent.Right() << std::endl;
        integ_dbg2 << "parent area = " << parent.Area();
        integ_dbg2 << " +- " << parent.Err() << std::endl;
        std::vector<IntRegion<Arg> > children;
        parent.SubDivide(&children);
        // For "GKP", there are only two, but for GKPOSC, there is one
        // for each oscillation in region

        // Try to do at least 3x better with the children
        Arg factor = 3 * children.size() * finalerr / tolerance;
        Arg newepsabs = fabs(parent.Err() / factor);
        Arg newepsrel = newepsabs / fabs(parent.Area());
        integ_dbg2 << "New epsabs, rel = " << newepsabs << ", " << newepsrel;
        integ_dbg2 << "  (" << children.size() << " children)\n";

        Arg newarea = Arg(0.0);
        Arg newerror = 0.0;
        for (size_t i = 0; i < children.size(); i++) {
            IntRegion<Arg> &child = children[i];
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

        Arg delta = parent.Area() - newarea;
        if (newerror <= parent.Err() && fabs(delta) <= parent.Err() && newerror >= 0.99 * parent.Err()) {
            integ_dbg2 << "roundoff type 1: delta/newarea = ";
            integ_dbg2 << fabs(delta) / fabs(newarea);
            integ_dbg2 << ", newerror/error = " << newerror / parent.Err() << std::endl;
            roundoffType1++;
        }
        if (iteration >= 10 && newerror > parent.Err() && fabs(delta) <= newerror - parent.Err()) {
            integ_dbg2 << "roundoff type 2: newerror/error = ";
            integ_dbg2 << newerror / parent.Err() << std::endl;
            roundoffType2 += std::min(newerror / parent.Err() - 1.0, Arg(1.0));
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
        IntRegion<Arg> const &r = allregions.top();
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
template <typename UnaryFunctionT>
struct AuxFunc1 { // f(1/x-1) for int(a..infinity)
public:
    AuxFunc1(UnaryFunctionT const &f) : _f(f) {}
    template <typename Arg>
    auto operator()(Arg x) const {
        return _f(1.0 / x - 1.0) / (x * x);
    }
private:
    UnaryFunctionT const &_f;
};

/**
 * Auxiliary function 1
 *
 */
template <class UF>
AuxFunc1<UF> inline Aux1(UF uf) {
    return AuxFunc1<UF>(uf);
}

template <typename UnaryFunctionT>
struct AuxFunc2 { // f(1/x+1) for int(-infinity..b)
public:
    AuxFunc2(UnaryFunctionT const &f) : _f(f) {}
    template <typename Arg>
    auto operator()(Arg x) const {
        return _f(1.0 / x + 1.0) / (x * x);
    }
private:
    UnaryFunctionT const &_f;
};

/**
 * Auxiliary function 2
 *
 */
template <class UF>
AuxFunc2<UF> inline Aux2(UF uf) {
    return AuxFunc2<UF>(uf);
}


// pulled from MoreFunctional.h.  Needed in class Int2DAuxType and Int3DAuxType
template <class BF>
class binder2_1 {
public:
    binder2_1(const BF &oper, typename BF::first_argument_type val) : _oper(oper), _value(val) {}
    typename BF::result_type operator()(const typename BF::second_argument_type &x) const {
        return _oper(_value, x);
    }

protected:
    BF _oper;
    typename BF::first_argument_type _value;
};

template <class BF, class Tp>
inline binder2_1<BF> bind21(const BF &oper, const Tp &x) {
    using Arg = typename BF::first_argument_type;
    return binder2_1<BF>(oper, static_cast<Arg>(x));
}

template <class BF, class YREG>
class Int2DAuxType {
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
class binder3_1 {
public:
    binder3_1(const TF &oper, typename TF::firstof3_argument_type val) : _oper(oper), _value(val) {}
    typename TF::result_type operator()(typename TF::secondof3_argument_type const &x1,
                                        typename TF::thirdof3_argument_type const &x2) const {
        return _oper(_value, x1, x2);
    }

protected:
    TF _oper;
    typename TF::firstof3_argument_type _value;
};

template <class TF, class Tp>
inline binder3_1<TF> bind31(const TF &oper, const Tp &x) {
    using Arg = typename TF::firstof3_argument_type;
    return binder3_1<TF>(oper, static_cast<Arg>(x));
}

template <class TF, class YREG, class ZREG>
class Int3DAuxType {
public:
    Int3DAuxType(const TF &func, const YREG &yreg, const ZREG &zreg, const typename TF::result_type &abserr,
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
    const TF &_func;
    const YREG &_yreg;
    const ZREG &_zreg;
    typename TF::result_type _abserr, _relerr;
};

}  // end namespace details

/**
 * Front end for the 1d integrator
 */
template <typename UnaryFunctionT, typename Arg>
inline Arg int1d(UnaryFunctionT func, IntRegion<Arg> &reg,
		      Arg const &abserr = DEFABSERR,
		      Arg const &relerr = DEFRELERR) {
    using namespace details;

    integ_dbg2 << "start int1d: " << reg.Left() << ".." << reg.Right() << std::endl;

    if ((reg.Left() <= -MOCK_INF && reg.Right() > 0) || (reg.Right() >= MOCK_INF && reg.Left() < 0)) {
        reg.AddSplit(0);
    }

    if (reg.NSplit() > 0) {
        std::vector<IntRegion<Arg> > children;
        reg.SubDivide(&children);
        integ_dbg2 << "Subdivided into " << children.size() << " children\n";
        Arg answer = Arg();
        Arg err = 0;
        for (size_t i = 0; i < children.size(); i++) {
            IntRegion<Arg> &child = children[i];
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
            IntRegion<Arg> modreg(1.0 / (reg.Right() - 1.0), 0.0, reg.getDbgout());
            intGKP(Aux2<UnaryFunctionT>(func), modreg, abserr, relerr);
            reg.SetArea(modreg.Area(), modreg.Err());
        } else if (reg.Right() >= MOCK_INF) {
            integ_dbg2 << "left = " << reg.Left() << ", right = infinity\n";
            assert(reg.Left() >= 0.0);
            IntRegion<Arg> modreg(0.0, 1.0 / (reg.Left() + 1.0), reg.getDbgout());
            intGKP(Aux1<UnaryFunctionT>(func), modreg, abserr, relerr);
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

// =============================================================
/**
 * The 1D integrator
 *
 * @note This simply wraps the int1d function above and handles the
 *       instantiation of the intRegion.
 *
 */
template <typename UnaryFunctionT, typename Arg>
auto integrate(UnaryFunctionT func,
               Arg const a,
               Arg const b,
               double eps = 1.0e-6) {
    IntRegion<Arg> region(a, b);
    return int1d(func, region, DEFABSERR, eps);
}

// =============================================================
/**
 * The 2D integrator
 *
 */
template <typename BinaryFunctionT, typename X, typename Y>
auto integrate2d(BinaryFunctionT func, X x1, X x2, Y y1, Y y2, double eps = 1.0e-6) {
    auto outer = [func, x1, x2, eps](auto y) {
        auto inner = [func, y](auto x) { return func(x, y); };
        return integrate(inner, x1, x2, eps);
    };
    return integrate(outer, y1, y2, eps);
}
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
