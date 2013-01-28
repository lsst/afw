#include "lsst/daf/base.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/image/XYTransform.h"
#include "lsst/afw/detection/Psf.h"

#ifndef LSST_AFW_DETECTION_WARPEDPSF_H
#define LSST_AFW_DETECTION_WARPEDPSF_H

namespace lsst {
namespace afw {
namespace detection {

//
// WarpedPsf: a class which combines an unwarped psf and a camera distortion
//
// If K_0(x,x') is the unwarped PSF, and f is the camera distortion, then the 
// warped PSF is defined by
//
//   K(f(x),f(x')) = K_0(x,x')      (*)
//
// We linearize the camera distortion in the vicinity of the point where the
// PSF is computed.  The definition (*) does not include the Jacobian of the
// transformation, since the afw convention is that PSF's are normalized to
// have integral 1 anyway.
//
// Note: In order to plug into a WarpedPsf, the undistorted Psf only needs to define
// the virtuals clone() and doGetLocalKernel().
//
class WarpedPsf : public Psf {
public:
    typedef boost::shared_ptr<WarpedPsf> Ptr;
    typedef boost::shared_ptr<const WarpedPsf> ConstPtr;

    typedef lsst::afw::image::Color Color;
    typedef lsst::afw::math::Kernel Kernel;
    typedef lsst::afw::geom::Point2I Point2I;
    typedef lsst::afw::geom::Point2D Point2D;
    typedef lsst::afw::geom::Extent2I Extent2I;
    typedef lsst::afw::image::XYTransform XYTransform;

    //
    // If p is the nominal pixel position, and p' is the true position on the sky, then our
    // convention for the transform is that p' = distortion.forwardTransform(p)
    //
    WarpedPsf(Psf::Ptr undistorted_psf, XYTransform::Ptr distortion);

protected:
    //
    // Devirtualize class Psf
    //
    // We currently don't define doGetKernel() (the default implementation in the parent class
    // returns a null pointer, which caller is responsible for handling).  Defining doGetKernel()
    // would be problematic; we would need to compute a "global" kernel size and center but these
    // are pixel-dependent.
    //
    virtual Psf::Ptr clone() const;

    //
    // API notes:
    //   (1) 'size' param can be Extent2I(0,0) if caller wants "native" size
    //   (2) 'distort' param ignored for now (this will eventually be removed in favor of a different API)
    //
    virtual Image::Ptr doComputeImage(Color const& color, Point2D const& ccdXY, 
				      Extent2I const& size, bool normalizePeak, bool distort) const;

    virtual Kernel::Ptr doGetLocalKernel(Point2D const &p, Color const &c)
    {
	return this->_doGetLocalKernel(p, c);
    }

    virtual Kernel::ConstPtr doGetLocalKernel(Point2D const &p, Color const &c) const
    {
	return this->_doGetLocalKernel(p, c);
    }

protected:
    Psf::Ptr _undistorted_psf;
    lsst::afw::image::XYTransform::Ptr _distortion;

    Kernel::Ptr _doGetLocalKernel(Point2D const &p, Color const &c) const;
    
    // the image returned by this member function is used in doComputeImage() and doGetLocalKernel()
    Image::Ptr _make_warped_kernel_image(Point2D const &p, Color const &color, Point2I &ctr) const;
};


}}}

#endif  // LSST_AFW_DETECTION_WARPEDPSF_H
