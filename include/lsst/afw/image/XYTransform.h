/*
 * XYTransform: Virtual base class representing an invertible transform of a pixelized image 
 *
 * TODO: serialization
 */

#ifndef LSST_AFW_IMAGE_XYTRANSFORM_H
#define LSST_AFW_IMAGE_XYTRANSFORM_H

#include <string>
#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/cameraGeom/Detector.h"

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
    typedef lsst::afw::geom::ellipses::Quadrupole Quadrupole;
    typedef lsst::afw::geom::AffineTransform AffineTransform;

    XYTransform(bool in_fp_coordinate_system);
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

    // apply distortion to an (infinitesimal) quadrupole
    Quadrupole forwardTransform(Point2D const &pixel, Quadrupole const &q) const;
    Quadrupole reverseTransform(Point2D const &pixel, Quadrupole const &q) const;

    bool in_fp_coordinate_system() const { return _in_fp_coordinate_system; }

protected:
    bool _in_fp_coordinate_system;
};


//
// IdentityXYTransform: Represents a trivial XYTransform satisfying f(x)=x.
// 
class IdentityXYTransform : public XYTransform
{
public:
    IdentityXYTransform(bool in_fp_coordinate_system);
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


//
// Note: RadialXYTransform is always in the FP coordinate system
//
class RadialXYTransform : public XYTransform
{
public:
    RadialXYTransform(std::vector<double> const &coeffs, bool coefficientsDistort=true);
    virtual ~RadialXYTransform() { }

    virtual PTR(XYTransform) clone() const;
    virtual PTR(XYTransform) invert() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &pixel) const;

    //
    // The following static member functions operate on polynomials represented by vector<double>.
    //
    // They are intended mainly as helpers for the virtual member functions above, but are declared
    // public since there are also some unit tests which call them.
    //
    static std::vector<double>  polyInvert(std::vector<double> const &coeffs);
    static double               polyEval(std::vector<double> const &coeffs, double x);
    static Point2D              polyEval(std::vector<double> const &coeffs, Point2D const &p);
    static double               polyEvalDeriv(std::vector<double> const &coeffs, double x);
    static AffineTransform      polyEvalJacobian(std::vector<double> const &coeffs, Point2D const &p);
    static double               polyEvalInverse(std::vector<double> const &coeffs, std::vector<double> const &icoeffs, double x);
    static Point2D              polyEvalInverse(std::vector<double> const &coeffs, std::vector<double> const &icoeffs, Point2D const &p);
    static AffineTransform      polyEvalInverseJacobian(std::vector<double> const &coeffs, std::vector<double> const &icoeffs, Point2D const &p);
    static AffineTransform      makeAffineTransform(double x, double y, double f, double g);

protected:
    std::vector<double> _coeffs;
    std::vector<double> _icoeffs;
    bool _coefficientsDistort;
};


class DetectorXYTransform : public XYTransform
{
public:
    typedef lsst::afw::cameraGeom::Detector Detector;

    DetectorXYTransform(CONST_PTR(XYTransform) fp_transform, CONST_PTR(Detector) detector);
    virtual ~DetectorXYTransform() { }

    virtual PTR(XYTransform) clone() const;
    virtual PTR(XYTransform) invert() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;

protected:
    CONST_PTR(XYTransform) _fp_transform;
    CONST_PTR(Detector) _detector;
};


}}}

#endif
