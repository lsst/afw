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

    // returns a deep copy
    virtual Ptr clone() const = 0;

    // returns a "deep inverse" in this sense that the forward+inverse transforms do not share state
    virtual PTR(XYTransform) invert() const;

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
// IdentityXYTransform: Represents a trivial XYTransform satisfying f(x)=x.
// 
class IdentityXYTransform : public XYTransform
{
public:
    IdentityXYTransform() { }
    virtual ~IdentityXYTransform() { }
    
    virtual PTR(XYTransform) clone() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    virtual lsst::afw::geom::AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual lsst::afw::geom::AffineTransform linearizeReverseTransform(Point2D const &pixel) const;
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

    XYTransformFromWcsPair(CONST_PTR(Wcs) dst, CONST_PTR(Wcs) src);
    virtual ~XYTransformFromWcsPair() { }

    virtual PTR(XYTransform) invert() const;

    // The following methods are needed to devirtualize the XYTransform parent class
    virtual XYTransform::Ptr clone() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    
protected:
    CONST_PTR(Wcs) _dst;
    CONST_PTR(Wcs) _src;
};  


//
// This guy is used to supply a default ->invert() method which works for any XYTransform
// (but can be overridden if something more efficient exists)
//
class InvertedXYTransform : public XYTransform
{
public:
    InvertedXYTransform(PTR(XYTransform) base);
    virtual ~InvertedXYTransform() { }

    virtual PTR(XYTransform) clone() const;
    virtual PTR(XYTransform) invert() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    virtual lsst::afw::geom::AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual lsst::afw::geom::AffineTransform linearizeReverseTransform(Point2D const &pixel) const;

protected:    
    PTR(XYTransform) _base;
};


}}}

#endif
