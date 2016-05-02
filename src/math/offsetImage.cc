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
 
/**
 * @file
 *
 * Offset an Image (or Mask or MaskedImage) by a constant vector (dx, dy)
 */
#include <iterator>
#include "lsst/afw/math/offsetImage.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/math/warpExposure.h"

namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;

namespace lsst {
namespace afw {
namespace math {

/**
 * @brief Return an image offset by (dx, dy) using the specified algorithm
 *
 * @note The image pixels are always offset by a fraction of a pixel and the image origin (XY0)
 * picks is modified to handle the integer portion of the offset.
 * In the special case that the offset in both x and y lies in the range (-1, 1) the origin is not changed.
 * Otherwise the pixels are shifted by (-0.5, 0.5] pixels and the origin shifted accordingly.
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if the algorithm is invalid
 */
template<typename ImageT>
typename ImageT::Ptr offsetImage(ImageT const& inImage,  ///< The %image to offset
                                 float dx,               ///< move the %image this far in the column direction
                                 float dy,               ///< move the %image this far in the row direction
                                 std::string const& algorithmName,  ///< Type of resampling Kernel to use
                                 unsigned int buffer ///< Width of buffer (border) around kernel image
                                    ///< to allow for warping edge effects (pixels).
                                    ///< Values < 0 are treated as 0.
                                    ///< This is only used during computation; the final image
                                    ///< has the same dimensions as the kernel.
                                ) {
    SeparableKernel::Ptr offsetKernel = makeWarpingKernel(algorithmName);

    typename ImageT::Ptr buffImage;
    if (buffer > 0) {
        // Paste input image into buffered image
        afwGeom::Extent2I const &dims = inImage.getDimensions();
        typename ImageT::Ptr buffered(new ImageT(dims.getX() + 2 * buffer, dims.getY() + 2 * buffer));
        buffImage = buffered;
        afwGeom::Box2I box(afwGeom::Point2I(buffer, buffer), dims);
        buffImage->assign(inImage, box);
    } else {
        buffImage = std::make_shared<ImageT>(inImage);
    }

    if (offsetKernel->getWidth() > buffImage->getWidth() || 
        offsetKernel->getHeight() > buffImage->getHeight()) {
        throw LSST_EXCEPT(pexExcept::LengthError,
                          (boost::format("Image of size %dx%d is too small to offset using a %s kernel"
                                         "(minimum %dx%d)") %
                           buffImage->getWidth() % buffImage->getHeight() % algorithmName %
                           offsetKernel->getWidth() % offsetKernel->getHeight()).str());
    }

//    typename ImageT::Ptr convImage(new ImageT(buffImage, true)); // output image, a deep copy
    typename ImageT::Ptr convImage(new ImageT(buffImage->getDimensions())); // Convolved image

    int dOrigX, dOrigY;
    double fracX, fracY;
    // If the offset in both axes is in (-1, 1) use it as is, and don't shift the origin
    if (dx > -1 && dx < 1 && dy > -1 && dy < 1) {
        dOrigX = 0;
        dOrigY = 0;
        fracX = dx;
        fracY = dy;
    } else {
        dOrigX = static_cast<int>(std::floor(dx + 0.5));
        dOrigY = static_cast<int>(std::floor(dy + 0.5));
        fracX = dx - dOrigX;
        fracY = dy - dOrigY;
    }

    // We seem to have to pass -fracX, -fracY to setKernelParameters, for reasons RHL doesn't understand
    double dKerX = -fracX;
    double dKerY = -fracY;

    //
    // If the shift is -ve, the generated shift kernel (e.g. Lanczos5) is quite asymmetric, with the
    // largest coefficients to the left of centre.  We therefore move the centre of calculated shift kernel
    // one to the right to center up the largest coefficients
    //
    if (dKerX < 0) {
        offsetKernel->setCtrX(offsetKernel->getCtrX() + 1);
    }
    if (dKerY < 0) {
        offsetKernel->setCtrY(offsetKernel->getCtrY() + 1);
    }
    
    offsetKernel->setKernelParameters(std::make_pair(dKerX, dKerY));

    convolve(*convImage, *buffImage, *offsetKernel, true, true);

    typename ImageT::Ptr outImage;
    if (buffer > 0) {
        afwGeom::Box2I box(afwGeom::Point2I(buffer, buffer), inImage.getDimensions());
        typename ImageT::Ptr out(new ImageT(*convImage, box, afwImage::LOCAL, true));
        outImage = out;
    } else {
        outImage = convImage;
    }

    // adjust the origin; do this after convolution since convolution also sets XY0
    outImage->setXY0(geom::Point2I(inImage.getX0() + dOrigX, inImage.getY0() + dOrigY));

    return outImage;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
/// \cond
#define INSTANTIATE(TYPE) \
    template afwImage::Image<TYPE>::Ptr offsetImage(afwImage::Image<TYPE> const&, float, float, \
                                                    std::string const&, unsigned int); \
    template afwImage::MaskedImage<TYPE>::Ptr offsetImage(afwImage::MaskedImage<TYPE> const&, float, float, \
                                                          std::string const&, unsigned int);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(int)
/// \endcond

}}}
