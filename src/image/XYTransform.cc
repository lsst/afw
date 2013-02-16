#include "boost/make_shared.hpp"
#include "lsst/afw/image/XYTransform.h"

namespace pexEx = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace image {


// -------------------------------------------------------------------------------------------------
//
// XYTransform


XYTransform::XYTransform(bool inFpCoordinateSystem) 
    : daf::base::Citizen(typeid(this)), _inFpCoordinateSystem(inFpCoordinateSystem)
{ }


/// default implementation; subclass may override
geom::AffineTransform XYTransform::linearizeForwardTransform(Point2D const &p) const
{
    Point2D px = p + geom::Extent2D(1,0);
    Point2D py = p + geom::Extent2D(0,1);

    return geom::makeAffineTransformFromTriple(p, px, py, 
                                                  this->forwardTransform(p),
                                                  this->forwardTransform(px), 
                                                  this->forwardTransform(py));
}


/// default implementation; subclass may override
geom::AffineTransform XYTransform::linearizeReverseTransform(Point2D const &p) const
{
    Point2D px = p + geom::Extent2D(1,0);
    Point2D py = p + geom::Extent2D(0,1);

    return geom::makeAffineTransformFromTriple(p, px, py, 
                                                  this->reverseTransform(p),
                                                  this->reverseTransform(px), 
                                                  this->reverseTransform(py));
}


/// default implementation; subclass may override
PTR(XYTransform) XYTransform::invert() const
{
    return boost::make_shared<InvertedXYTransform> (this->clone());
}

geom::ellipses::Quadrupole XYTransform::forwardTransform(Point2D const &pixel, Quadrupole const &q) const
{
    // Note: q.transform(L) returns (LQL^T)
    AffineTransform a = linearizeForwardTransform(pixel);
    return q.transform(a.getLinear());
}

geom::ellipses::Quadrupole XYTransform::reverseTransform(Point2D const &pixel, Quadrupole const &q) const
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

geom::Point2D IdentityXYTransform::forwardTransform(Point2D const &pixel) const
{
    return pixel;
}

geom::Point2D IdentityXYTransform::reverseTransform(Point2D const &pixel) const
{
    return pixel;
}

geom::AffineTransform IdentityXYTransform::linearizeForwardTransform(Point2D const &pixel) const
{
    // note: AffineTransform constructor called with no arguments gives the identity transform
    return geom::AffineTransform(); 
}

geom::AffineTransform IdentityXYTransform::linearizeReverseTransform(Point2D const &pixel) const
{
    // note: AffineTransform constructor called with no arguments gives the identity transform
    return geom::AffineTransform(); 
}


// -------------------------------------------------------------------------------------------------
//
// XYTransformFromWcsPair


XYTransformFromWcsPair::XYTransformFromWcsPair(CONST_PTR(Wcs) dst, CONST_PTR(Wcs) src)
    : XYTransform(false), _dst(dst), _src(src)
{ }


PTR(XYTransform) XYTransformFromWcsPair::clone() const
{
    return boost::make_shared<XYTransformFromWcsPair>(_dst->clone(), _src->clone());
}


geom::Point2D XYTransformFromWcsPair::forwardTransform(Point2D const &pixel) const
{
    //
    // TODO there is an alternate version of pixelToSky() which is designated for the 
    // "knowledgeable user in need of performance".  This is probably better, but first I need 
    // to understand exactly which checks are needed (e.g. I think we need to check by hand 
    // that both Wcs's use the same celestial coordinate system)
    //
    PTR(afw::coord::Coord) x = _src->pixelToSky(pixel);
    return _dst->skyToPixel(*x);
}

geom::Point2D XYTransformFromWcsPair::reverseTransform(Point2D const &pixel) const
{
    PTR(afw::coord::Coord) x = _dst->pixelToSky(pixel);
    return _src->skyToPixel(*x);
}

PTR(XYTransform) XYTransformFromWcsPair::invert() const
{
    // clone and swap src,dst
    return boost::make_shared<XYTransformFromWcsPair> (_src->clone(), _dst->clone());
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

geom::Point2D InvertedXYTransform::forwardTransform(Point2D const &pixel) const
{
    return _base->reverseTransform(pixel);
}

geom::Point2D InvertedXYTransform::reverseTransform(Point2D const &pixel) const
{
    return _base->forwardTransform(pixel);
}

geom::AffineTransform InvertedXYTransform::linearizeForwardTransform(Point2D const &pixel) const
{
    return _base->linearizeReverseTransform(pixel);
}

geom::AffineTransform InvertedXYTransform::linearizeReverseTransform(Point2D const &pixel) const
{
    return _base->linearizeForwardTransform(pixel);
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

geom::Point2D RadialXYTransform::forwardTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEval(_coeffs,p) : polyEvalInverse(_coeffs,_icoeffs,p);
}

geom::Point2D RadialXYTransform::reverseTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEvalInverse(_coeffs,_icoeffs,p) : polyEval(_coeffs,p);
}

geom::AffineTransform RadialXYTransform::linearizeForwardTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEvalJacobian(_coeffs,p) 
        : polyEvalInverseJacobian(_coeffs,_icoeffs,p);
}

geom::AffineTransform RadialXYTransform::linearizeReverseTransform(Point2D const &p) const
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

geom::Point2D RadialXYTransform::polyEval(std::vector<double> const &coeffs, Point2D const &p)
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

geom::AffineTransform 
RadialXYTransform::polyEvalJacobian(std::vector<double> const &coeffs, Point2D const &p)
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

geom::Point2D RadialXYTransform::polyEvalInverse(std::vector<double> const &coeffs, 
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

geom::AffineTransform 
RadialXYTransform::polyEvalInverseJacobian(std::vector<double> const &coeffs, 
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

geom::AffineTransform 
RadialXYTransform::makeAffineTransform(double x, double y, double rnew, double rderiv)
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


// -----------------------------------------------------------------------------------------------------------
//
// DetectorXYTransform


DetectorXYTransform::DetectorXYTransform(CONST_PTR(XYTransform) fpTransform, 
                                         CONST_PTR(Detector) detector)
    : XYTransform(false), _fpTransform(fpTransform), _detector(detector)
{
    if (!fpTransform->inFpCoordinateSystem()) {
        throw LSST_EXCEPT(pexEx::InvalidParameterException, 
                          "DetectorXYTransform base must be in FP coordinate system");
    }
}

PTR(XYTransform) DetectorXYTransform::clone() const
{
    // note: there is no detector->clone()
    return boost::make_shared<DetectorXYTransform> (_fpTransform->clone(), _detector);
}

PTR(XYTransform) DetectorXYTransform::invert() const
{
    return boost::make_shared<DetectorXYTransform> (_fpTransform->invert(), _detector);
}

geom::Point2D DetectorXYTransform::forwardTransform(Point2D const &p) const
{
    Point2D q;
    q = _detector->getPositionFromPixel(p).getMm();
    q = _fpTransform->forwardTransform(q);
    q = _detector->getPixelFromPosition(FpPoint(q));
    return q;
}

geom::Point2D DetectorXYTransform::reverseTransform(Point2D const &p) const
{
    Point2D q;
    q = _detector->getPositionFromPixel(p).getMm();
    q = _fpTransform->reverseTransform(q);
    q = _detector->getPixelFromPosition(FpPoint(q));
    return q;
}

geom::AffineTransform DetectorXYTransform::linearizeForwardTransform(Point2D const &p) const
{
    AffineTransform a;
    a = _detector->linearizePositionFromPixel(p);
    a = _fpTransform->linearizeForwardTransform(a(p)) * a;
    a = _detector->linearizePixelFromPosition(FpPoint(a(p))) * a;
    return a;
}

geom::AffineTransform DetectorXYTransform::linearizeReverseTransform(Point2D const &p) const
{
    AffineTransform a;
    a = _detector->linearizePositionFromPixel(p);
    a = _fpTransform->linearizeReverseTransform(a(p)) * a;
    a = _detector->linearizePixelFromPosition(FpPoint(a(p))) * a;
    return a;    
}


}}}
