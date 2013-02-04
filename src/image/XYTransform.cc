#include "boost/make_shared.hpp"
//#include "lsst/daf/base/PropertySet.h"
//#include "lsst/daf/base/PropertyList.h"
//#include "lsst/pex/exceptions.h"
//#include "lsst/afw/geom/Point.h"
//#include "lsst/afw/geom/AffineTransform.h"
//#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/XYTransform.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwCG = lsst::afw::cameraGeom;
namespace pexEx = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace image {


// -------------------------------------------------------------------------------------------------
//
// XYTransform


XYTransform::XYTransform(bool in_fp_coordinate_system) 
    : daf::base::Citizen(typeid(this)), _in_fp_coordinate_system(in_fp_coordinate_system)
{ }


//
// Static helper method: returns the unique AffineTransform such that
//   (p0,p1) -> q
//   (p0+1,p1) -> qx
//   (p0,p1+1) -> qy
// where caller specifies p=(p0,p1) and q,qx,qy
//
// This implementation could be made faster, but it seems unlikely this will ever be a bottleneck
//
afwGeom::AffineTransform XYTransform::finiteDifference(Point2D const &p, Point2D const &q, Point2D const &qx, Point2D const &qy)
{
    double p0 = p.getX();
    double p1 = p.getY();

    double q0 = q.getX();
    double q1 = q.getY();

    double qx0 = qx.getX();
    double qx1 = qx.getY();

    double qy0 = qy.getX();
    double qy1 = qy.getY();

    Eigen::Matrix3d mp;
    mp <<  p0,  p0+1,   p0,
	   p1,    p1, p1+1,
	  1.0,   1.0,  1.0;

    Eigen::Matrix3d mq;
    mq <<  q0,  qx0,  qy0,
	   q1,  qx1,  qy1,
	  1.0,  1.0,  1.0;

    Eigen::Matrix3d m = mq * mp.inverse();
    return AffineTransform(m);
}


// default implementation; subclass may override
afwGeom::AffineTransform XYTransform::linearizeForwardTransform(Point2D const &p) const
{
    return finiteDifference(p, this->forwardTransform(p), 
			    this->forwardTransform(p + afwGeom::Extent2D(1,0)),
			    this->forwardTransform(p + afwGeom::Extent2D(0,1)));
}


// default implementation; subclass may override
afwGeom::AffineTransform XYTransform::linearizeReverseTransform(Point2D const &p) const
{
    return finiteDifference(p, this->reverseTransform(p), 
			    this->reverseTransform(p + afwGeom::Extent2D(1,0)),
			    this->reverseTransform(p + afwGeom::Extent2D(0,1)));
}


// default implementation; subclass may override
PTR(XYTransform) XYTransform::invert() const
{
    return boost::make_shared<InvertedXYTransform> (this->clone());
}


// -------------------------------------------------------------------------------------------------
//
// IdentityXYTransform


IdentityXYTransform::IdentityXYTransform(bool in_fp_coordinate_system)
    : XYTransform(in_fp_coordinate_system)
{ }

PTR(XYTransform) IdentityXYTransform::clone() const
{
    return boost::make_shared<IdentityXYTransform> (_in_fp_coordinate_system);
}

lsst::afw::geom::Point2D IdentityXYTransform::forwardTransform(Point2D const &pixel) const
{
    return pixel;
}

lsst::afw::geom::Point2D IdentityXYTransform::reverseTransform(Point2D const &pixel) const
{
    return pixel;
}

lsst::afw::geom::AffineTransform IdentityXYTransform::linearizeForwardTransform(Point2D const &pixel) const
{
    // note: AffineTransform constructor called with no arguments gives the identity transform
    return lsst::afw::geom::AffineTransform(); 
}

lsst::afw::geom::AffineTransform IdentityXYTransform::linearizeReverseTransform(Point2D const &pixel) const
{
    // note: AffineTransform constructor called with no arguments gives the identity transform
    return lsst::afw::geom::AffineTransform(); 
}


// -------------------------------------------------------------------------------------------------
//
// XYTransformFromWcsPair


XYTransformFromWcsPair::XYTransformFromWcsPair(CONST_PTR(Wcs) dst, CONST_PTR(Wcs) src)
    : XYTransform(false), _dst(dst), _src(src)
{ }


XYTransform::Ptr XYTransformFromWcsPair::clone() const
{
    return boost::make_shared<XYTransformFromWcsPair>(_dst->clone(), _src->clone());
}


afwGeom::Point2D XYTransformFromWcsPair::forwardTransform(Point2D const &pixel) const
{
    //
    // TODO there is an alternate version of pixelToSky() which is designated for the "knowledgeable user
    // in need of performance".  This is probably better, but first I need to understand exactly which checks
    // are needed (e.g. I think we need to check by hand that both Wcs's use the same celestial coordinate
    // system)
    //
    lsst::afw::coord::Coord::Ptr x = _dst->pixelToSky(pixel);
    return _src->skyToPixel(*x);
}

afwGeom::Point2D XYTransformFromWcsPair::reverseTransform(Point2D const &pixel) const
{
    lsst::afw::coord::Coord::Ptr x = _src->pixelToSky(pixel);
    return _dst->skyToPixel(*x);
}

PTR(XYTransform) XYTransformFromWcsPair::invert() const
{
    // clone and swap src,dst
    return boost::make_shared<XYTransformFromWcsPair> (_src->clone(), _dst->clone());
}


// -------------------------------------------------------------------------------------------------
//
// InvertedXYTransform


InvertedXYTransform::InvertedXYTransform(PTR(XYTransform) base)
    : XYTransform(base->in_fp_coordinate_system()), _base(base)
{ }

PTR(XYTransform) InvertedXYTransform::clone() const
{
    // deep copy
    return boost::make_shared<InvertedXYTransform> (_base->clone());
}

PTR(XYTransform) InvertedXYTransform::invert() const
{
    return _base;
}

afwGeom::Point2D InvertedXYTransform::forwardTransform(Point2D const &pixel) const
{
    return _base->reverseTransform(pixel);
}

afwGeom::Point2D InvertedXYTransform::reverseTransform(Point2D const &pixel) const
{
    return _base->forwardTransform(pixel);
}

afwGeom::AffineTransform InvertedXYTransform::linearizeForwardTransform(Point2D const &pixel) const
{
    return _base->linearizeReverseTransform(pixel);
}

afwGeom::AffineTransform InvertedXYTransform::linearizeReverseTransform(Point2D const &pixel) const
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
        throw LSST_EXCEPT(pexEx::InvalidParameterException, "invalid parameters for radial distortion");
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

afwGeom::Point2D RadialXYTransform::forwardTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEval(_coeffs,p) : polyEvalInverse(_coeffs,_icoeffs,p);
}

afwGeom::Point2D RadialXYTransform::reverseTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEvalInverse(_coeffs,_icoeffs,p) : polyEval(_coeffs,p);
}

afwGeom::AffineTransform RadialXYTransform::linearizeForwardTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEvalJacobian(_coeffs,p) : polyEvalInverseJacobian(_coeffs,_icoeffs,p);
}

afwGeom::AffineTransform RadialXYTransform::linearizeReverseTransform(Point2D const &p) const
{
    return _coefficientsDistort ? polyEvalInverseJacobian(_coeffs,_icoeffs,p) : polyEvalJacobian(_coeffs,p);
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
        throw LSST_EXCEPT(pexEx::InvalidParameterException, "invalid parameters in RadialXYTransform::polyInvert");

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

afwGeom::Point2D RadialXYTransform::polyEval(std::vector<double> const &coeffs, Point2D const &p)
{
    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x+y*y);

    if (r > 0.0) {
        double rnew = polyEval(coeffs,r);
        return Point2D(rnew*x/r, rnew*y/r);
    }

    if (coeffs.size() == 0 || coeffs[0] != 0.0)
        throw LSST_EXCEPT(pexEx::InvalidParameterException, "invalid parameters for radial distortion");

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

afwGeom::AffineTransform RadialXYTransform::polyEvalJacobian(std::vector<double> const &coeffs, Point2D const &p)
{
    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x+y*y);
    double rnew = polyEval(coeffs,r);
    double rderiv = polyEvalDeriv(coeffs,r);
    return makeAffineTransform(x, y, rnew, rderiv);
}

double RadialXYTransform::polyEvalInverse(std::vector<double> const &coeffs, std::vector<double> const &icoeffs, double x)
{
    static const int maxIter = 10;
    double tolerance = 1.0e-14 * x;

    double r = polyEval(icoeffs, x);      // initial guess
    int iter = 0;
    
    for (;;) {
        double dx = x - polyEval(coeffs,r);   // residual
        if (fabs(dx) <= tolerance)
            return r;
        if (iter++ > maxIter)
            throw LSST_EXCEPT(pexEx::RuntimeErrorException, "max iteration count exceeded in RadialXYTransform::polyEvalInverse");
        r += dx / polyEvalDeriv(coeffs,r);   // Newton-Raphson iteration
    }
}

afwGeom::Point2D RadialXYTransform::polyEvalInverse(std::vector<double> const &coeffs, std::vector<double> const &icoeffs, Point2D const &p)
{
    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x+y*y);

    if (r > 0.0) {
        double rnew = polyEvalInverse(coeffs, icoeffs, r);
        return Point2D(rnew*x/r, rnew*y/r);
    }

    if (coeffs.size() == 0 || coeffs[0] != 0.0)
        throw LSST_EXCEPT(pexEx::InvalidParameterException, "invalid parameters for radial distortion");

    return Point2D(0,0);    
}

afwGeom::AffineTransform RadialXYTransform::polyEvalInverseJacobian(std::vector<double> const &coeffs, std::vector<double> const &icoeffs, Point2D const &p)
{
    double x = p.getX();
    double y = p.getY();
    double r = sqrt(x*x+y*y);
    double rnew = polyEvalInverse(coeffs,icoeffs,r);
    double rderiv = 1.0 / polyEvalDeriv(coeffs,rnew);
    return makeAffineTransform(x, y, rnew, rderiv);
}

afwGeom::AffineTransform RadialXYTransform::makeAffineTransform(double x, double y, double rnew, double rderiv)
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
    // Propagating through the formulas below, the AffineTransform is [rderiv*I + O(r) + O(10^{-17})]
    // which is fine (assuming rderiv is nonzero as r->0).
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


DetectorXYTransform::DetectorXYTransform(CONST_PTR(XYTransform) fp_transform, CONST_PTR(Detector) detector)
    : XYTransform(false), _fp_transform(fp_transform), _detector(detector)
{
    if (!fp_transform->in_fp_coordinate_system()) {
        throw LSST_EXCEPT(pexEx::InvalidParameterException, 
                          "first argument to DetectorXYTransform constructor not in FP coordinate system");
    }
}

PTR(XYTransform) DetectorXYTransform::clone() const
{
    // note: there is no detector->clone()
    return boost::make_shared<DetectorXYTransform> (_fp_transform->clone(), _detector);
}

PTR(XYTransform) DetectorXYTransform::invert() const
{
    return boost::make_shared<DetectorXYTransform> (_fp_transform->invert(), _detector);
}

afwGeom::Point2D DetectorXYTransform::forwardTransform(Point2D const &p) const
{
    Point2D q;
    q = _detector->getPositionFromPixel(p).getMm();
    q = _fp_transform->forwardTransform(q);
    q = _detector->getPixelFromPosition(afwCG::FpPoint(q));
    return q;
}

afwGeom::Point2D DetectorXYTransform::reverseTransform(Point2D const &p) const
{
    Point2D q;
    q = _detector->getPositionFromPixel(p).getMm();
    q = _fp_transform->reverseTransform(q);
    q = _detector->getPixelFromPosition(afwCG::FpPoint(q));
    return q;
}


}}}
