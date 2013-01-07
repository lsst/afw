// #include <boost/make_shared.hpp>
#include "lsst/afw/detection/WarpedPsf.h"
#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/math/detail/SrcPosFunctor.h"

namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

typedef afwGeom::Point2D Point2D;

namespace lsst {
namespace afw {
namespace detection {


// -------------------------------------------------------------------------------------------------


static inline double min4(double a, double b, double c, double d)
{
    return std::min(std::min(a,b), std::min(c,d));
}

static inline double max4(double a, double b, double c, double d)
{
    return std::max(std::max(a,b), std::max(c,d));
}

static inline afwGeom::AffineTransform getLinear(const afwGeom::AffineTransform &a)
{
    afwGeom::AffineTransform ret;
    ret[0] = a[0];
    ret[1] = a[1];
    ret[2] = a[2];
    ret[3] = a[3];
    ret[4] = 0.0;
    ret[5] = 0.0;
    return ret;
}

//
// This helper function is essentially an alternate interface to afw::math::warpImage()
// in which the caller does not need to precompute the output bounding box.
//
// We preserve the convention of warpImage() that the affine transform is inverted,
// so that the output and input images are related by:
//   out[p] = in[A^{-1}p]
//
// The input image is assumed zero-padded.
//
static inline Psf::Image::Ptr warpAffine(Psf::Image const &im, afwGeom::AffineTransform const &t)
{
    //
    // hmmm, are these the best choices?
    //
    static const char *interpolation_name = "lanczos5";
    static const int interpolation_padding = 0;

    // min/max coordinate values in input image
    int in_xlo = im.getX0();
    int in_xhi = im.getX0() + im.getWidth() - 1;
    int in_ylo = im.getY0();
    int in_yhi = im.getY0() + im.getHeight() - 1;

    // corners of output image
    Point2D c00 = t(Point2D(in_xlo,in_ylo));
    Point2D c01 = t(Point2D(in_xlo,in_yhi));
    Point2D c10 = t(Point2D(in_xhi,in_ylo));
    Point2D c11 = t(Point2D(in_xhi,in_yhi));

    //
    // bounding box for output image
    //
    int out_xlo = floor(min4(c00.getX(),c01.getX(),c10.getX(),c11.getX())) - interpolation_padding;
    int out_ylo = floor(min4(c00.getY(),c01.getY(),c10.getY(),c11.getY())) - interpolation_padding;
    int out_xhi = ceil(max4(c00.getX(),c01.getX(),c10.getX(),c11.getX())) + interpolation_padding;
    int out_yhi = ceil(max4(c00.getY(),c01.getY(),c10.getY(),c11.getY())) + interpolation_padding;

    // allocate output image
    Psf::Image::Ptr ret = boost::make_shared<Psf::Image>(out_xhi-out_xlo+1, out_yhi-out_ylo+1);
    ret->setXY0(afwGeom::Point2I(out_xlo,out_ylo));

    // warp it!
    afwMath::WarpingControl wc(interpolation_name);
    afwMath::warpImage(*ret, im, t, wc);
    return ret;
}


// -------------------------------------------------------------------------------------------------


WarpedPsf::WarpedPsf(Psf::Ptr undistorted_psf, XYTransform::Ptr distortion)
{
    _undistorted_psf = undistorted_psf;
    _distortion = distortion;
}

Psf::Ptr WarpedPsf::clone() const
{
    return boost::make_shared<WarpedPsf>(_undistorted_psf->clone(), _distortion->clone());
}

Psf::Image::Ptr WarpedPsf::doComputeImage(Color const& color, Point2D const& ccdXY, Extent2I const& size, bool normalizePeak, bool distort) const
{
    Point2I ctr;
    PTR(Image) im = this->_make_warped_kernel_image(ccdXY, color, ctr);
    
    int width = (size.getX() > 0) ? size.getX() : im->getWidth();
    int height = (size.getY() > 0) ? size.getY() : im->getHeight();

    if ((width != im->getWidth()) || (height != im->getHeight())) {
	PTR(Image) im2 = boost::make_shared<Image> (width, height);
	ctr = resizeKernelImage(*im2, *im, ctr);
	im = im2;
    }

    if (normalizePeak) {
	double centralPixelValue = (*im)(ctr.getX(), ctr.getY());
	*im /= centralPixelValue;
    }

    return recenterKernelImage(im, ctr, ccdXY);
}

afwMath::Kernel::Ptr WarpedPsf::_doGetLocalKernel(Point2D const &p, Color const &c) const
{
    Point2I ctr;
    PTR(Image) im = this->_make_warped_kernel_image(p, c, ctr);
    PTR(afwMath::Kernel) ret = boost::make_shared<afwMath::FixedKernel>(*im);
    ret->setCtr(ctr);
    return ret;
}

//
// The image returned by this routine is a "kernel image", i.e. xy0 is not meaningful
// but there is a distinguished central pixel which corresponds to the point "p" where
// the PSF is evaluated.
//
// FIXME for now, the image returned by this routine is normalized to sum 1, following
// the convention in the parent Psf class.  This convention seems fishy to me and I'll
// revisit it later...
//
Psf::Image::Ptr WarpedPsf::_make_warped_kernel_image(Point2D const &p, Color const &c, Point2I &ctr) const
{
    afwGeom::AffineTransform t = _distortion->linearizeReverseTransform(p);
    Point2D tp = t(p);

    Kernel::Ptr k = _undistorted_psf->getLocalKernel(tp, c);
    PTR(Image) im = boost::make_shared<Image>(k->getWidth(), k->getHeight());
    k->computeImage(*im, true, tp.getX(), tp.getY());

    //
    // im->xy0 is undefined (im is a "kernel image"); set it appropriately
    // for a coordinate system with 'tp' at the origin
    //
    im->setXY0(Point2I(-k->getCtrX(), -k->getCtrY()));

    // Go to the warped coordinate system with 'p' at the origin
    Psf::Image::Ptr ret = warpAffine(*im, getLinear(t.invert()));

    // ret->xy0 is meaningful, but for consistency with the kernel API, we use a parallel Point2I instead
    ctr = Point2I(-ret->getX0(), -ret->getY0());

    // 
    // Normalize the output image to sum 1
    // FIXME defining a member function Image::getSum() would be convenient here and in other places
    //
    double imSum = 0.0;
    for (int y = 0; y != ret->getHeight(); ++y) {
	Image::x_iterator imEnd = ret->row_end(y);
	for (Image::x_iterator imPtr = ret->row_begin(y); imPtr != imEnd; imPtr++) {
            imSum += *imPtr;
        }
    }
    if (imSum == 0.0) {
	throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "psf image has sum 0");
    }
    *ret /= imSum;

    return ret;
}


}}}

