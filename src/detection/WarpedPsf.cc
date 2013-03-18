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

#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/WarpedPsf.h"
#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/math/detail/SrcPosFunctor.h"

namespace lsst { namespace afw { namespace detection {

static inline double min4(double a, double b, double c, double d) {
    return std::min(std::min(a,b), std::min(c,d));
}

static inline double max4(double a, double b, double c, double d) {
    return std::max(std::max(a,b), std::max(c,d));
}

static inline geom::AffineTransform getLinear(const geom::AffineTransform &a) {
    geom::AffineTransform ret;
    ret[0] = a[0];
    ret[1] = a[1];
    ret[2] = a[2];
    ret[3] = a[3];
    ret[4] = 0.0;
    ret[5] = 0.0;
    return ret;
}

// TODO: make this routine externally callable and more generic using templates
//  (also useful in e.g. math/offsetImage.cc)
static inline PTR(Psf::Image) zeroPadImage(Psf::Image const &im, int pad)
{
    int nx = im.getWidth();
    int ny = im.getHeight();

    PTR(Psf::Image) out = boost::make_shared<Psf::Image> (nx+2*pad, ny+2*pad);
    out->setXY0(im.getX0()-pad, im.getY0()-pad);

    geom::Box2I box(geom::Point2I(pad,pad), geom::Extent2I(nx,ny));
    PTR(Psf::Image) subimage = boost::make_shared<Psf::Image> (*out, box);
    *subimage <<= im;

    return out;
}

/**
 * @brief Alternate interface to afw::math::warpImage()
 * in which the caller does not need to precompute the output bounding box.
 *
 * We preserve the convention of warpImage() that the affine transform is inverted,
 * so that the output and input images are related by:
 *   out[p] = in[A^{-1}p]
 *
 * The input image is assumed zero-padded.
 */
static inline PTR(Psf::Image) warpAffine(Psf::Image const &im, geom::AffineTransform const &t)
{
    //
    // hmmm, are these the best choices?
    //
    static const char *interpolation_name = "lanczos5";
    static const int dst_padding = 0;
    static const int src_padding = 5;

    // min/max coordinate values in input image
    int in_xlo = im.getX0();
    int in_xhi = im.getX0() + im.getWidth() - 1;
    int in_ylo = im.getY0();
    int in_yhi = im.getY0() + im.getHeight() - 1;

    // corners of output image
    geom::Point2D c00 = t(geom::Point2D(in_xlo,in_ylo));
    geom::Point2D c01 = t(geom::Point2D(in_xlo,in_yhi));
    geom::Point2D c10 = t(geom::Point2D(in_xhi,in_ylo));
    geom::Point2D c11 = t(geom::Point2D(in_xhi,in_yhi));

    //
    // bounding box for output image
    //
    int out_xlo = floor(min4(c00.getX(),c01.getX(),c10.getX(),c11.getX())) - dst_padding;
    int out_ylo = floor(min4(c00.getY(),c01.getY(),c10.getY(),c11.getY())) - dst_padding;
    int out_xhi = ceil(max4(c00.getX(),c01.getX(),c10.getX(),c11.getX())) + dst_padding;
    int out_yhi = ceil(max4(c00.getY(),c01.getY(),c10.getY(),c11.getY())) + dst_padding;

    // allocate output image
    PTR(Psf::Image) ret = boost::make_shared<Psf::Image>(out_xhi-out_xlo+1, out_yhi-out_ylo+1);
    ret->setXY0(geom::Point2I(out_xlo,out_ylo));

    // zero-pad input image
    PTR(Psf::Image) im_padded = zeroPadImage(im, src_padding);

    // warp it!
    math::WarpingControl wc(interpolation_name);
    math::warpImage(*ret, *im_padded, t, wc, 0.0);
    return ret;
}

WarpedPsf::WarpedPsf(CONST_PTR(Psf) undistorted_psf, CONST_PTR(XYTransform) distortion) {
    if (distortion->inFpCoordinateSystem()) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterException, 
                          "WarpedPsf constructor: distortion must not be in FP coordinate system");
    }

    _undistorted_psf = undistorted_psf;
    _distortion = distortion;
    if (!_undistorted_psf) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Undistorted Psf passed to WarpedPsf must not be None/NULL"
        );
    }
    if (!_distortion) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "XYTransform passed to WarpedPsf must not be None/NULL"
        );
    }
}

PTR(Psf) WarpedPsf::clone() const {
    return boost::make_shared<WarpedPsf>(_undistorted_psf->clone(), _distortion->clone());
}

PTR(Psf::Image) WarpedPsf::doComputeKernelImage(
    geom::Point2D const & ccdXY, image::Color const & color
) const {
    geom::AffineTransform t = _distortion->linearizeReverseTransform(ccdXY);
    geom::Point2D tp = t(ccdXY);

    PTR(Image) im = _undistorted_psf->computeKernelImage(tp, color);

    // Go to the warped coordinate system with 'p' at the origin
    PTR(Psf::Image) ret = warpAffine(*im, getLinear(t.invert()));

    double normFactor = 1.0;
    // 
    // Normalize the output image to sum 1
    // FIXME defining a member function Image::getSum() would be convenient here and in other places
    //
    normFactor = 0.0;
    for (int y = 0; y != ret->getHeight(); ++y) {
        Image::x_iterator imEnd = ret->row_end(y);
        for (Image::x_iterator imPtr = ret->row_begin(y); imPtr != imEnd; imPtr++) {
            normFactor += *imPtr;
        }
    }
    if (normFactor == 0.0) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterException, "psf image has sum 0");
    }
    *ret /= normFactor;
    return ret;
}

}}} // namepace lsst::afw::detection
