// -*- LSST-C++ -*-

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
 * @brief Contains AssertDimensionsOK function
 *
 * @author Kresimir Cosic
 * @author Original code by Russell Owen
 *
 * @note Extraction to this file requested by RHL during code review
 *
 * @ingroup afw
 */

#include <cstdint>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/detail/Convolve.h"
#include "lsst/afw/math/detail/ConvolveShared.h"

namespace afwImage = lsst::afw::image;
namespace mathDetail = lsst::afw::math::detail;

/*
 * Assert that the dimensions of convolvedImage, inImage and kernel are compatible with convolution.
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dim.
 * @throw lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or h.
 * @throw lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 *
 * @note Same as assertDimensionsOK in basicConvolve.cc, copy-pasted
 */
template <typename OutImageT, typename InImageT>
void mathDetail::assertDimensionsOK(
    OutImageT const &convolvedImage,
    InImageT const &inImage,
    lsst::afw::math::Kernel const &kernel
) {
    if (convolvedImage.getDimensions() != inImage.getDimensions()) {
        std::ostringstream os;
        os << "convolvedImage dimensions = ( "
        << convolvedImage.getWidth() << ", " << convolvedImage.getHeight()
        << ") != (" << inImage.getWidth() << ", " << inImage.getHeight() << ") = inImage dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    if (inImage.getWidth() < kernel.getWidth() || inImage.getHeight() < kernel.getHeight()) {
        std::ostringstream os;
        os << "inImage dimensions = ( "
        << inImage.getWidth() << ", " << inImage.getHeight()
        << ") smaller than (" << kernel.getWidth() << ", " << kernel.getHeight()
        << ") = kernel dimensions in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    if ((kernel.getWidth() < 1) || (kernel.getHeight() < 1)) {
        std::ostringstream os;
        os << "kernel dimensions = ( "
        << kernel.getWidth() << ", " << kernel.getHeight()
        << ") smaller than (1, 1) in width and/or height";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
}

/*
 * Explicit instantiation
 */
/// \cond
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define NL /* */
// Instantiate Image or MaskedImage versions
#define INSTANTIATE_IM_OR_MI(IMGMACRO, OUTPIXTYPE, INPIXTYPE) \
    template void mathDetail::assertDimensionsOK(     \
        IMGMACRO(OUTPIXTYPE) const &convolvedImage,   \
        IMGMACRO(INPIXTYPE) const &inImage,           \
        lsst::afw::math::Kernel const &kernel         \
                                       );

// Instantiate both Image and MaskedImage versions
#define INSTANTIATE(OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(IMAGE,       OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(MASKEDIMAGE, OUTPIXTYPE, INPIXTYPE)

INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, std::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, std::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(std::uint16_t, std::uint16_t)
/// \endcond
