/*
 * XYTransform: Virtual base class representing an invertible transform of a pixelized image 
 *
 * TODO: serialization
 */

#ifndef LSST_AFW_IMAGE_XYTRANSFORM_H
#define LSST_AFW_IMAGE_XYTRANSFORM_H

namespace lsst {
namespace afw {
namespace image {


//
// A virtual base class which represents a "pixel domain to pixel domain" transform (e.g. camera distortion)
// By comparison, class Wcs represents a "pixel domain to celestial" transform.
//
// Right now, there is almost nothing here, but it will be expanded later!
//
class XYTransform : public lsst::daf::base::Citizen
{
public:
    typedef boost::shared_ptr<lsst::afw::image::XYTransform> Ptr;
    typedef boost::shared_ptr<lsst::afw::image::XYTransform const> ConstPtr;
    typedef lsst::afw::geom::Point2D Point2D;
    typedef lsst::afw::geom::AffineTransform AffineTransform;

    XYTransform();
    virtual ~XYTransform() { }

    // note: should be a deep copy
    virtual Ptr clone() const = 0;

    // in general, the XYTransform returned by this routine may share state with the original
    // XYTransform; use invert()->clone() to make them independent
    virtual PTR(XYTransform) invert() const = 0;

    //
    // Both the @pixel argument and the return value of these routines:
    //   - are in pixel units
    //   - have XY0 offsets included (i.e. caller may need to add XY0 to @pixel 
    //          and subtract XY0 from return value if necessary)
    //
    // These routines are responsible for throwing exceptions if the 'pixel' arg is outside the domain of the transform.
    //
    virtual Point2D forwardTransform(Point2D const &pixel) const = 0;
    virtual Point2D reverseTransform(Point2D const &pixel) const = 0;
    
    //
    // These guys are virtual but not pure virtual; there is a default implementation which
    // calls forwardTransform() or reverseTransform() and takes finite differences with step 
    // size equal to one pixel.
    //
    // The following should always be satisfied (and analogously for the reverse transform)
    //    this->forwardTransform(p) == this->linearizeForwardTransform(p)(p);   // where p is an arbitrary Point2D
    //
    virtual lsst::afw::geom::AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual lsst::afw::geom::AffineTransform linearizeReverseTransform(Point2D const &pixel) const;

    // helper function which may be useful elsewhere; see XYTransform.cc
    static lsst::afw::geom::AffineTransform finiteDifference(Point2D const &p, Point2D const &q, Point2D const &qx, Point2D const &qy);
};


//
// XYTransformFromWcsPair: Represents an XYTransform obtained by putting two Wcs's "back to back".
//
// Eventually there will be an XYTransform subclass which represents a camera distortion.
// For now we can get a SIP camera distortion in a clunky way, by using an XYTransformFromWcsPair
// with a SIP-distorted TanWcs and an undistorted Wcs.
//
// Note: this is very similar to class lsst::afw::math::detail::WcsSrcPosFunctor
//   but watch out since the XY0 offset convention is different!!
//
class XYTransformFromWcsPair : public XYTransform
{
public:
    typedef boost::shared_ptr<XYTransformFromWcsPair> Ptr;
    typedef boost::shared_ptr<XYTransformFromWcsPair const> ConstPtr;
    typedef lsst::afw::image::Wcs Wcs;

    XYTransformFromWcsPair(Wcs::Ptr dst, Wcs::Ptr src);
    virtual ~XYTransformFromWcsPair() { }

    // The following methods are needed to devirtualize the XYTransform parent class
    virtual XYTransform::Ptr clone() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    virtual PTR(XYTransform) invert() const;
    
protected:
    Wcs::Ptr _dst;
    Wcs::Ptr _src;
};  


}}}

#endif
