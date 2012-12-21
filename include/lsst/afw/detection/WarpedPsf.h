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
// If B_0(x) = true surface brightness (unwarped, not PSF convolved)
//    B_1 = unwarped PSF-convolved brightness
//    B_2 = warped PSF-convolved brightness
//
//    P_0 = unwarped PSF = convolution kernel relating B_0 and B_1
//    P_1 = warped PSF = convolution kernel relating B_0 and B_2
//
// Then
//   B_2(x) = B_1(T(x))
//          = 
//
//   B_{obs}(x) = int d^2x' P(x,x') B_0(x')
//
//   B_
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
    // The main work is done in doComputeImage().  We also define doGetLocalKernel(), which simply calls doComputeImage()
    // and returns a FixedKernel.  We currently don't define doGetKernel() (the default implementation in the parent class
    // returns a null pointer, which caller is responsible for treating as an error).  Defining doGetKernel() is problematic:
    // we would need to compute a "global" kernel size; this is tricky since the size has pixel dependence which is controlled
    // by derivatives of the distortion.
    //
    virtual Psf::Ptr clone() const;

    //
    // Notes:
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

    Image::Ptr _make_warped_kernel_image(Point2D const &p, Color const &color, Point2I &ctr) const;
    Kernel::Ptr _doGetLocalKernel(Point2D const &p, Color const &c) const;
};


}}}

#endif  // LSST_AFW_DETECTION_WARPEDPSF_H
