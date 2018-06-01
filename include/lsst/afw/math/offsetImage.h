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

#if !defined(LSST_AFW_MATH_OFFSETIMAGE_H)
#define LSST_AFW_MATH_OFFSETIMAGE_H 1

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

namespace lsst {
namespace afw {
namespace math {

/**
 * Return an image offset by (dx, dy) using the specified algorithm
 *
 * @param image The %image to offset
 * @param dx move the %image this far in the column direction
 * @param dy move the %image this far in the row direction
 * @param algorithmName Type of resampling Kernel to use
 * @param buffer Width of buffer (border) around kernel image to allow for warping edge
 *               effects (pixels). Values < 0 are treated as 0. This is only used during
 *               computation; the final image has the same dimensions as the kernel.
 *
 * @note The image pixels are always offset by a fraction of a pixel and the image origin (XY0)
 * picks is modified to handle the integer portion of the offset.
 * In the special case that the offset in both x and y lies in the range (-1, 1) the origin is not changed.
 * Otherwise the pixels are shifted by (-0.5, 0.5] pixels and the origin shifted accordingly.
 *
 * @throws lsst::pex::exceptions::InvalidParameterError if the algorithm is invalid
 */
template <typename ImageT>
std::shared_ptr<ImageT> offsetImage(ImageT const& image, float dx, float dy,
                                    std::string const& algorithmName = "lanczos5", unsigned int buffer = 0);
/**
 * Rotate an image by an integral number of quarter turns
 *
 * @param image The %image to rotate
 * @param nQuarter the desired number of quarter turns
 */
template <typename ImageT>
std::shared_ptr<ImageT> rotateImageBy90(ImageT const& image, int nQuarter);

/**
 * Flip an image left--right and/or top--bottom
 *
 * @param inImage The %image to flip
 * @param flipLR Flip left <--> right?
 * @param flipTB Flip top <--> bottom?
 */
template <typename ImageT>
std::shared_ptr<ImageT> flipImage(ImageT const& inImage, bool flipLR, bool flipTB);
/**
 * @param inImage The %image to bin
 * @param binX Output pixels are binX*binY input pixels
 * @param binY Output pixels are binX*binY input pixels
 * @param flags how to generate super-pixels
 */
template <typename ImageT>
std::shared_ptr<ImageT> binImage(ImageT const& inImage, int const binX, int const binY,
                                 lsst::afw::math::Property const flags = lsst::afw::math::MEAN);
/**
 * @param inImage The %image to bin
 * @param binsize Output pixels are binsize*binsize input pixels
 * @param flags how to generate super-pixels
 */
template <typename ImageT>
std::shared_ptr<ImageT> binImage(ImageT const& inImage, int const binsize,
                                 lsst::afw::math::Property const flags = lsst::afw::math::MEAN);
}  // namespace math
}  // namespace afw
}  // namespace lsst
#endif
