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

#include "lsst/afw/geom/XYTransform.h"
#include "boost/make_shared.hpp"

namespace pexEx = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace geom {


// -------------------------------------------------------------------------------------------------
//
// XYTransform


XYTransform::XYTransform(bool inFpCoordinateSystem) 
    : daf::base::Citizen(typeid(this)), _inFpCoordinateSystem(inFpCoordinateSystem)
{ }


/// default implementation; subclass may override
AffineTransform XYTransform::linearizeForwardTransform(Point2D const &p) const
{
    Point2D px = p + Extent2D(1,0);
    Point2D py = p + Extent2D(0,1);

    return makeAffineTransformFromTriple(p, px, py, 
                                                  this->forwardTransform(p),
                                                  this->forwardTransform(px), 
                                                  this->forwardTransform(py));
}


/// default implementation; subclass may override
AffineTransform XYTransform::linearizeReverseTransform(Point2D const &p) const
{
    Point2D px = p + Extent2D(1,0);
    Point2D py = p + Extent2D(0,1);

    return makeAffineTransformFromTriple(p, px, py, 
                                                  this->reverseTransform(p),
                                                  this->reverseTransform(px), 
                                                  this->reverseTransform(py));
}


/// default implementation; subclass may override
PTR(XYTransform) XYTransform::invert() const
{
    return boost::make_shared<InvertedXYTransform> (this->clone());
}

ellipses::Quadrupole XYTransform::forwardTransform(Point2D const &pixel, Quadrupole const &q) const
{
    // Note: q.transform(L) returns (LQL^T)
    AffineTransform a = linearizeForwardTransform(pixel);
    return q.transform(a.getLinear());
}

ellipses::Quadrupole XYTransform::reverseTransform(Point2D const &pixel, Quadrupole const &q) const
{
    AffineTransform a = linearizeReverseTransform(pixel);
    return q.transform(a.getLinear());
}



// -------------------------------------------------------------------------------------------------
//
// IdentityXYTransform


IdentityXYTransform::IdentityXYTransform(bool inFpCoordinateSystem)
    : XYTransform(inFpCoordinateSystem)
{ }

PTR(XYTransform) IdentityXYTransform::clone() const
{
    return boost::make_shared<IdentityXYTransform> (_inFpCoordinateSystem);
}

Point2D IdentityXYTransform::forwardTransform(Point2D const &pixel) const
{
    return pixel;
}

Point2D IdentityXYTransform::reverseTransform(Point2D const &pixel) const
{
    return pixel;
}

AffineTransform IdentityXYTransform::linearizeForwardTransform(Point2D const &pixel) const
{
    // note: AffineTransform constructor called with no arguments gives the identity transform
    return AffineTransform(); 
}

AffineTransform IdentityXYTransform::linearizeReverseTransform(Point2D const &pixel) const
{
    // note: AffineTransform constructor called with no arguments gives the identity transform
    return AffineTransform(); 
}


// -------------------------------------------------------------------------------------------------
//
// InvertedXYTransform


InvertedXYTransform::InvertedXYTransform(CONST_PTR(XYTransform) base)
    : XYTransform(base->inFpCoordinateSystem()), _base(base)
{ }

PTR(XYTransform) InvertedXYTransform::clone() const
{
    // deep copy
    return boost::make_shared<InvertedXYTransform> (_base->clone());
}

PTR(XYTransform) InvertedXYTransform::invert() const
{
    return _base->clone();
}

Point2D InvertedXYTransform::forwardTransform(Point2D const &pixel) const
{
    return _base->reverseTransform(pixel);
}

Point2D InvertedXYTransform::reverseTransform(Point2D const &pixel) const
{
    return _base->forwardTransform(pixel);
}

AffineTransform InvertedXYTransform::linearizeForwardTransform(Point2D const &pixel) const
{
    return _base->linearizeReverseTransform(pixel);
}

AffineTransform InvertedXYTransform::linearizeReverseTransform(Point2D const &pixel) const
{
    return _base->linearizeForwardTransform(pixel);
}

// -------------------------------------------------------------------------------------------------
//
// AffineXYTransform
/*
 * @brief This wraps an AffineTransform as an XYTransform for inclusing in a TransformRegistry
 * Note that the inFpCoordinateSystem is not used and set to false
 *
 */

AffineXYTransform::AffineXYTransform(AffineTransform const &affineTransform)
    : XYTransform(false), _forwardAffineTransform(affineTransform), 
      _reverseAffineTransform(_forwardAffineTransform.invert())
{ }

PTR(XYTransform) AffineXYTransform::clone() const
{
    return boost::make_shared<AffineXYTransform> (_forwardAffineTransform);
}

Point2D AffineXYTransform::forwardTransform(Point2D const &position) const
{
    return _forwardAffineTransform(position);
}

Point2D AffineXYTransform::reverseTransform(Point2D const &position) const
{
    return _reverseAffineTransform(position);
}

AffineTransform AffineXYTransform::getForwardTransform() const
{
    return _forwardAffineTransform;
}

AffineTransform AffineXYTransform::getReverseTransform() const
{
    return _reverseAffineTransform;
}

AffineTransform AffineXYTransform::linearizeForwardTransform(Point2D const &pixel) const
{
    return _forwardAffineTransform;
}

AffineTransform AffineXYTransform::linearizeReverseTransform(Point2D const &pixel) const
{
    return _reverseAffineTransform; 
}


// -------------------------------------------------------------------------------------------------
//
// RadialXYTransform


RadialXYTransform::RadialXYTransform(std::vector<double> const &coeffs, bool coefficientsDistort)
    : XYTransform(true)
{
    if (coeffs.size() == 0) {
        // constructor called with no arguments = identity transformation
        _coeffs.resize(2);
        _coeffs[0] = 0.0;
        _coeffs[1] = 1.0;
    }
    else if ((coeffs.size() == 1) || (coeffs[0] != 0.0) || (coeffs[1] == 0.0)) {
        // Discontinuous or singular transformation; presumably unintentional so throw exception
        throw LSST_EXCEPT(pexEx::InvalidParameterException, 
                          "invalid parameters for radial distortion");
    }
    else {
        _coeffs = coeffs;
    }

    _icoeffs = polyInvert(_coeffs);
    _coefficientsDistort = coefficientsDistort;
}

PTR(XYTransform) RadialXYTransform::clone() const
{
    return boost::make_shared<RadialXYTransform> (_coeffs, _coefficientsDistort);    
}

PTR(XYTransform) RadialXYTransform::invert() const
{
    return boost::make_shared<RadialXYTransform> (_coeffs, !_coefficientsDistort);
}

Point2D RadialXYTransform::forwardTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEval(_coeffs,p) : polyEvalInverse(_coeffs,_icoeffs,p);
}

Point2D RadialXYTransform::reverseTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEvalInverse(_coeffs,_icoeffs,p) : polyEval(_coeffs,p);
}

AffineTransform RadialXYTransform::linearizeForwardTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEvalJacobian(_coeffs,p) 
        : polyEvalInverseJacobian(_coeffs,_icoeffs,p);
}

AffineTransform RadialXYTransform::linearizeReverseTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEvalInverseJacobian(_coeffs,_icoeffs,p) 
        : polyEvalJacobian(_coeffs,p);
}


// --- Note: all subsequent RadialXYTransform member functions are static

/*
 * @brief Invert the coefficients for the polynomial.
 *
 * We'll need the coeffs for the inverse of the input polynomial
 * handle up to 6th order
 * terms from Mathematical Handbook of Formula's and Tables, Spiegel & Liu.
 * This is a taylor approx, so not perfect.  We'll use it to get close to the inverse
 * and then use Newton-Raphson to get to machine precision. (only needs 1 or 2 iterations)
 */
std::vector<double> RadialXYTransform::polyInvert(std::vector<double> const &coeffs)
{
    static const unsigned int maxN = 7;   // degree of output polynomial + 1
    
    //
    // Some sanity checks.  The formulas for the inversion below assume c0 == 0 and c1 != 0
    //
    if (coeffs.size() <= 1 || coeffs.size() > maxN || coeffs[0] != 0.0 || coeffs[1] == 0.0)
        throw LSST_EXCEPT(pexEx::InvalidParameterException, 
                          "invalid parameters in RadialXYTransform::polyInvert");

    std::vector<double> c = coeffs;
    c.resize(maxN, 0.0);

    std::vector<double> ic(maxN);

    ic[0]  = 0.0;

    ic[1]  = 1.0;
    ic[1] /= c[1];

    ic[2]  = -c[2];
    ic[2] /= std::pow(c[1],3);

    ic[3]  = 2.0*c[2]*c[2] - c[1]*c[3];
    ic[3] /= std::pow(c[1],5);

    ic[4]  = 5.0*c[1]*c[2]*c[3] - 5.0*c[2]*c[2]*c[2] - c[1]*c[1]*c[4];
    ic[4] /= std::pow(c[1],7);
    
    ic[5]  = 6.0*c[1]*c[1]*c[2]*c[4] + 3.0*c[1]*c[1]*c[3]*c[3] - c[1]*c[1]*c[1]*c[5] + 
        14.0*std::pow(c[2], 4) - 21.0*c[1]*c[2]*c[2]*c[3];

    ic[6]  = 7.0*c[1]*c[1]*c[1]*c[2]*c[5] + 84.0*c[1]*c[2]*c[2]*c[2]*c[3] +
        7.0*c[1]*c[1]*c[1]*c[3]*c[4] - 28.0*c[1]*c[1]*c[2]*c[3]*c[3] - 
        std::pow(c[1], 4)*c[6] - 28.0*c[1]*c[1]*c[2]*c[2]*c[4] - 42.0*std::pow(c[2], 5);
    ic[6] /= std::pow(c[1],11);
    
    return ic;
}

double RadialXYTransform::polyEval(std::vector<double> const &coeffs, double x)
{
    int n = coeffs.size();

    double ret = 0.0;
    for (int i = n-1; i >= 0; i--)
        ret = ret*x + coeffs[i];

    return ret;
}

Point2D RadialXYTransform::polyEval(std::vector<double> const &coeffs, Point2D const &p)
{
    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x+y*y);

    if (r > 0.0) {
        double rnew = polyEval(coeffs,r);
        return Point2D(rnew*x/r, rnew*y/r);
    }

    if (coeffs.size() == 0 || coeffs[0] != 0.0) {
        throw LSST_EXCEPT(pexEx::InvalidParameterException, 
                          "invalid parameters for radial distortion");
    }

    return Point2D(0,0);
}

double RadialXYTransform::polyEvalDeriv(std::vector<double> const &coeffs, double x)
{
    int n = coeffs.size();

    double ret = 0.0;
    for (int i = n-1; i >= 1; i--)
        ret = ret*x + i*coeffs[i];

    return ret;
}

AffineTransform RadialXYTransform::polyEvalJacobian(std::vector<double> const &coeffs, Point2D const &p)
{
    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x+y*y);
    double rnew = polyEval(coeffs,r);
    double rderiv = polyEvalDeriv(coeffs,r);
    return makeAffineTransform(x, y, rnew, rderiv);
}

double RadialXYTransform::polyEvalInverse(std::vector<double> const &coeffs, 
                                          std::vector<double> const &icoeffs, double x)
{
    static const int maxIter = 10;
    double tolerance = 1.0e-14 * x;

    double r = polyEval(icoeffs, x);      // initial guess
    int iter = 0;
    
    for (;;) {
        double dx = x - polyEval(coeffs,r);   // residual
        if (fabs(dx) <= tolerance)
            return r;
        if (iter++ > maxIter) {
            throw LSST_EXCEPT(pexEx::RuntimeErrorException, 
                              "max iteration count exceeded in RadialXYTransform::polyEvalInverse");
        }
        r += dx / polyEvalDeriv(coeffs,r);   // Newton-Raphson iteration
    }
}

Point2D RadialXYTransform::polyEvalInverse(std::vector<double> const &coeffs, 
                                                    std::vector<double> const &icoeffs, 
                                                    Point2D const &p)
{
    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x+y*y);

    if (r > 0.0) {
        double rnew = polyEvalInverse(coeffs, icoeffs, r);
        return Point2D(rnew*x/r, rnew*y/r);
    }

    if (coeffs.size() == 0 || coeffs[0] != 0.0) {
        throw LSST_EXCEPT(pexEx::InvalidParameterException, 
                          "invalid parameters for radial distortion");
    }

    return Point2D(0,0);    
}

AffineTransform RadialXYTransform::polyEvalInverseJacobian(std::vector<double> const &coeffs, 
                                                           std::vector<double> const &icoeffs, 
                                                           Point2D const &p)
{
    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x+y*y);
    double rnew = polyEvalInverse(coeffs,icoeffs,r);
    double rderiv = 1.0 / polyEvalDeriv(coeffs,rnew);
    return makeAffineTransform(x, y, rnew, rderiv);
}

AffineTransform RadialXYTransform::makeAffineTransform(double x, double y, double rnew, double rderiv)
{
    double r = sqrt(x*x + y*y);
    
    if (r <= 0.0) {
        AffineTransform ret;
        ret[0] = ret[3] = rderiv;   // ret = rderiv * (identity)
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
    double t = rderiv - rnew/r;
    
    AffineTransform ret;
    ret[0] = (rderiv*x*x + rnew/r*y*y) / (r*r);    // a00
    ret[1] = ret[2] = t*x*y / (r*r);               // a01 == a10 for this transform
    ret[3] = (rderiv*y*y + rnew/r*x*x) / (r*r);    // a11
    ret[4] = -t*x;                                 // v0
    ret[5] = -t*y;                                 // v1
    return ret;
}


}}}
