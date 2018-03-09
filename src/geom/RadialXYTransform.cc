// -*- lsst-c++ -*-

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

#include <cmath>
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/XYTransform.h"
#include "lsst/afw/geom/Angle.h"
#include <memory>

namespace pexEx = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace geom {

RadialXYTransform::RadialXYTransform(std::vector<double> const &coeffs) : XYTransform() {
    if (coeffs.empty()) {
        // constructor called with no arguments = identity transformation
        _coeffs.resize(2);
        _coeffs[0] = 0.0;
        _coeffs[1] = 1.0;
    } else {
        if ((coeffs.size() == 1) || (coeffs[0] != 0.0) || (coeffs[1] == 0.0)) {
            // Discontinuous or singular transformation; presumably unintentional so throw exception
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "invalid parameters for radial distortion: need coeffs.size() != 1, "
                              "coeffs[0]==0, coeffs[1]!=0");
        }
        _coeffs = coeffs;
    }

    _icoeffs = polyInvert(_coeffs);
}

RadialXYTransform::RadialXYTransform(RadialXYTransform const &) = default;
RadialXYTransform::RadialXYTransform(RadialXYTransform &&) = default;
RadialXYTransform &RadialXYTransform::operator=(RadialXYTransform const &) = default;
RadialXYTransform &RadialXYTransform::operator=(RadialXYTransform &&) = default;
RadialXYTransform::~RadialXYTransform() = default;

std::shared_ptr<XYTransform> RadialXYTransform::clone() const {
    return std::make_shared<RadialXYTransform>(_coeffs);
}

std::shared_ptr<XYTransform> RadialXYTransform::invert() const {
    return std::make_shared<RadialXYTransform>(_coeffs);
}

Point2D RadialXYTransform::forwardTransform(Point2D const &p) const { return polyEval(_coeffs, p); }

Point2D RadialXYTransform::reverseTransform(Point2D const &p) const {
    return polyEvalInverse(_coeffs, _icoeffs, p);
}

AffineTransform RadialXYTransform::linearizeForwardTransform(Point2D const &p) const {
    return polyEvalJacobian(_coeffs, p);
}

AffineTransform RadialXYTransform::linearizeReverseTransform(Point2D const &p) const {
    return polyEvalInverseJacobian(_coeffs, _icoeffs, p);
}

// --- Note: all subsequent RadialXYTransform member functions are static

/**
 * @internal Invert the coefficients for the polynomial.
 *
 * We'll need the coeffs for the inverse of the input polynomial
 * handle up to 6th order
 * terms from Mathematical Handbook of Formula's and Tables, Spiegel & Liu.
 * This is a taylor approx, so not perfect.  We'll use it to get close to the inverse
 * and then use Newton-Raphson to get to machine precision. (only needs 1 or 2 iterations)
 */
std::vector<double> RadialXYTransform::polyInvert(std::vector<double> const &coeffs) {
    static const unsigned int maxN = 7;  // degree of output polynomial + 1

    //
    // Some sanity checks.  The formulas for the inversion below assume c0 == 0 and c1 != 0
    //
    if (coeffs.size() <= 1 || coeffs.size() > maxN || coeffs[0] != 0.0 || coeffs[1] == 0.0)
        throw LSST_EXCEPT(pexEx::InvalidParameterError,
                          "invalid parameters in RadialXYTransform::polyInvert");

    std::vector<double> c = coeffs;
    c.resize(maxN, 0.0);

    std::vector<double> ic(maxN);

    ic[0] = 0.0;

    ic[1] = 1.0;
    ic[1] /= c[1];

    ic[2] = -c[2];
    ic[2] /= std::pow(c[1], 3);

    ic[3] = 2.0 * c[2] * c[2] - c[1] * c[3];
    ic[3] /= std::pow(c[1], 5);

    ic[4] = 5.0 * c[1] * c[2] * c[3] - 5.0 * c[2] * c[2] * c[2] - c[1] * c[1] * c[4];
    ic[4] /= std::pow(c[1], 7);

    ic[5] = 6.0 * c[1] * c[1] * c[2] * c[4] + 3.0 * c[1] * c[1] * c[3] * c[3] - c[1] * c[1] * c[1] * c[5] +
            14.0 * std::pow(c[2], 4) - 21.0 * c[1] * c[2] * c[2] * c[3];
    ic[5] /= std::pow(c[1], 9);

    ic[6] = 7.0 * c[1] * c[1] * c[1] * c[2] * c[5] + 84.0 * c[1] * c[2] * c[2] * c[2] * c[3] +
            7.0 * c[1] * c[1] * c[1] * c[3] * c[4] - 28.0 * c[1] * c[1] * c[2] * c[3] * c[3] -
            std::pow(c[1], 4) * c[6] - 28.0 * c[1] * c[1] * c[2] * c[2] * c[4] - 42.0 * std::pow(c[2], 5);
    ic[6] /= std::pow(c[1], 11);

    return ic;
}

double RadialXYTransform::polyEval(std::vector<double> const &coeffs, double x) {
    int n = coeffs.size();

    double ret = 0.0;
    for (int i = n - 1; i >= 0; i--) ret = ret * x + coeffs[i];

    return ret;
}

Point2D RadialXYTransform::polyEval(std::vector<double> const &coeffs, Point2D const &p) {
    double r = p.asEigen().norm();

    if (r > 0.0) {
        double rnew = polyEval(coeffs, r);
        return Point2D(rnew * p.getX() / r, rnew * p.getY() / r);
    }

    if (coeffs.empty() || coeffs[0] != 0.0) {
        throw LSST_EXCEPT(pexEx::InvalidParameterError, "invalid parameters for radial distortion");
    }

    return Point2D(0, 0);
}

double RadialXYTransform::polyEvalDeriv(std::vector<double> const &coeffs, double x) {
    int n = coeffs.size();

    double ret = 0.0;
    for (int i = n - 1; i >= 1; i--) ret = ret * x + i * coeffs[i];

    return ret;
}

AffineTransform RadialXYTransform::polyEvalJacobian(std::vector<double> const &coeffs, Point2D const &p) {
    double r = p.asEigen().norm();
    double rnew = polyEval(coeffs, r);
    double rderiv = polyEvalDeriv(coeffs, r);
    return makeAffineTransform(p.getX(), p.getY(), rnew, rderiv);
}

double RadialXYTransform::polyEvalInverse(std::vector<double> const &coeffs,
                                          std::vector<double> const &icoeffs, double x) {
    static const int maxIter = 1000;
    double tolerance = 1.0e-14 * x;

    double r = polyEval(icoeffs, x);  // initial guess
    int iter = 0;

    for (;;) {
        double dx = x - polyEval(coeffs, r);  // residual
        if (fabs(dx) <= tolerance) return r;
        if (iter++ > maxIter) {
            throw LSST_EXCEPT(pexEx::RuntimeError,
                              "max iteration count exceeded in RadialXYTransform::polyEvalInverse");
        }
        r += dx / polyEvalDeriv(coeffs, r);  // Newton-Raphson iteration
    }
}

Point2D RadialXYTransform::polyEvalInverse(std::vector<double> const &coeffs,
                                           std::vector<double> const &icoeffs, Point2D const &p) {
    double r = p.asEigen().norm();

    if (r > 0.0) {
        double rnew = polyEvalInverse(coeffs, icoeffs, r);
        return Point2D(rnew * p.getX() / r, rnew * p.getY() / r);
    }

    if (coeffs.empty() || coeffs[0] != 0.0) {
        throw LSST_EXCEPT(pexEx::InvalidParameterError, "invalid parameters for radial distortion");
    }

    return Point2D(0, 0);
}

AffineTransform RadialXYTransform::polyEvalInverseJacobian(std::vector<double> const &coeffs,
                                                           std::vector<double> const &icoeffs,
                                                           Point2D const &p) {
    double r = p.asEigen().norm();
    double rnew = polyEvalInverse(coeffs, icoeffs, r);
    double rderiv = 1.0 / polyEvalDeriv(coeffs, rnew);
    return makeAffineTransform(p.getX(), p.getY(), rnew, rderiv);
}

AffineTransform RadialXYTransform::makeAffineTransform(double x, double y, double rnew, double rderiv) {
    double r = ::hypot(x, y);

    if (r <= 0.0) {
        AffineTransform ret;
        ret[0] = ret[3] = rderiv;  // ret = rderiv * (identity)
        return ret;
    }

    //
    // Note: calculation of "t" is numerically unstable as r->0, since p'(r) and p(r)/r will be
    // nearly equal.  However, detailed analysis shows that this is actually OK.  The numerical
    // instability means that the roundoff error in t is O(10^{-17}) even though t is formally O(r).
    //
    // Propagating through the formulas below, the AffineTransform is
    // [rderiv*I + O(r) + O(10^{-17})] which is fine (assuming rderiv is nonzero as r->0).
    //
    double t = rderiv - rnew / r;

    AffineTransform ret;
    ret[0] = (rderiv * x * x + rnew / r * y * y) / (r * r);  // a00
    ret[1] = ret[2] = t * x * y / (r * r);                   // a01 == a10 for this transform
    ret[3] = (rderiv * y * y + rnew / r * x * x) / (r * r);  // a11
    ret[4] = -t * x;                                         // v0
    ret[5] = -t * y;                                         // v1
    return ret;
}
}  // namespace geom
}  // namespace afw
}  // namespace lsst
