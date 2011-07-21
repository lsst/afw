


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

#ifndef LSST_AFW_MATH_DETAIL_GPU_CONVOLVE_H
#define LSST_AFW_MATH_DETAIL_GPU_CONVOLVE_H
/**
 * @file
 *
 * @brief Convolution support
 *
 * @author Kresimir Cosic
 * @author Original CPU convolution code by Russell Owen
 *
 * @ingroup afw
 */
#include <sstream>

#include "boost/shared_ptr.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/ConvolveImage.h"

#define IS_INSTANCE(A, B) (dynamic_cast<B const*>(&(A)) != NULL)


namespace lsst {
namespace afw {
namespace math {
namespace detail {
    template <typename OutImageT, typename InImageT>
    bool basicConvolveGPU(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            lsst::afw::math::Kernel const& kernel,
            lsst::afw::math::ConvolutionControl const& convolutionControl);

    template <typename OutPixelT, typename InPixelT>
    bool convolveLinearCombinationGPU(
            lsst::afw::image::MaskedImage<OutPixelT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>& convolvedImage,      ///< convolved %image
            lsst::afw::image::MaskedImage<InPixelT , lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> const& inImage,        ///< %image to convolve
            lsst::afw::math::LinearCombinationKernel const& kernel,
            lsst::afw::math::ConvolutionControl const& convolutionControl);

    template <typename OutPixelT, typename InPixelT>
    bool convolveLinearCombinationGPU(
            lsst::afw::image::Image<OutPixelT>& convolvedImage,      ///< convolved %image
            lsst::afw::image::Image<InPixelT > const& inImage,        ///< %image to convolve
            lsst::afw::math::LinearCombinationKernel const& kernel,
            lsst::afw::math::ConvolutionControl const& convolutionControl);

    template <typename OutPixelT, typename InPixelT>
    bool convolveSpatiallyInvariantGPU(
            lsst::afw::image::MaskedImage<OutPixelT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>& convolvedImage,      ///< convolved %image
            lsst::afw::image::MaskedImage<InPixelT , lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> const& inImage,        ///< %image to convolve
            lsst::afw::math::Kernel const& kernel,  ///< convolution kernel
            bool doNormalize);               ///< if true, normalize the kernel, else use "as is"

    template <typename OutPixelT, typename InPixelT>
    bool convolveSpatiallyInvariantGPU(
            lsst::afw::image::Image<OutPixelT>& convolvedImage,      ///< convolved %image
            lsst::afw::image::Image<InPixelT > const& inImage,        ///< %image to convolve
            lsst::afw::math::Kernel const& kernel,  ///< convolution kernel
            bool doNormalize);               ///< if true, normalize the kernel, else use "as is"
}
}
}
}



#endif // !defined(LSST_AFW_MATH_DETAIL_GPU_CONVOLVE_H)

