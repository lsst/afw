#include "boost/make_shared.hpp"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/XYTransform.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;

namespace lsst {
namespace afw {
namespace image {


// -------------------------------------------------------------------------------------------------
//
// XYTransform


XYTransform::XYTransform() 
    : daf::base::Citizen(typeid(this))
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
// XYTransformFromWcsPair


XYTransformFromWcsPair::XYTransformFromWcsPair(Wcs::Ptr dst, Wcs::Ptr src)
    : XYTransform(), _dst(dst), _src(src)
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
    : _base(base)
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


}}}
