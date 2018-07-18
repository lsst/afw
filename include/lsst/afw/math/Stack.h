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

#if !defined(LSST_AFW_MATH_STACK_H)
#define LSST_AFW_MATH_STACK_H
/*
 * Functions to stack images
 */
#include <vector>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/math/Statistics.h"

namespace lsst {
namespace afw {
namespace math {

/* ****************************************************************** *
 *
 * z stacks
 *
 * ******************************************************************* */

/**
 * A function to compute some statistics of a stack of Images
 */
template <typename PixelT>
std::shared_ptr<lsst::afw::image::Image<PixelT>> statisticsStack(
        std::vector<std::shared_ptr<lsst::afw::image::Image<PixelT>>>& images,  ///< Images to process
        Property flags,                                                         ///< statistics requested
        StatisticsControl const& sctrl = StatisticsControl(),                   ///< Control structure
        std::vector<lsst::afw::image::VariancePixel> const& wvector =
                std::vector<lsst::afw::image::VariancePixel>(0)  ///< vector containing weights
);

/**
 * @ brief compute statistical stack of Image.  Write to output image in-situ
 */
template <typename PixelT>
void statisticsStack(
        lsst::afw::image::Image<PixelT>& out,                                   ///< Output image
        std::vector<std::shared_ptr<lsst::afw::image::Image<PixelT>>>& images,  ///< Images to process
        Property flags,                                                         ///< statistics requested
        StatisticsControl const& sctrl = StatisticsControl(),                   ///< Control structure
        std::vector<lsst::afw::image::VariancePixel> const& wvector =
                std::vector<lsst::afw::image::VariancePixel>(0)  ///< vector containing weights
);

/**
 * A function to compute some statistics of a stack of Masked Images
 *
 * @param[in] images    MaskedImages to process.
 * @param[in] flags     Statistics requested.
 * @param[in] sctrl     Control structure.
 * @param[in] wvector   Vector of weights.
 * @param[in] clipped   Mask to set for pixels that were clipped (NOT rejected
 *                      due to masks).
 * @param[in] maskMap   Vector of pairs of mask pixel values; any pixel
 *                      on an input with any of the bits in .first will result
 *                      in all of the bits in .second being set on the
 *                      corresponding pixel on the output.
 *
 * If none of the input images are valid for some pixel,
 * the afwMath::StatisticsControl::getNoGoodPixelsMask() bit(s) are set.
 *
 * All the work is done in the function computeMaskedImageStack.
 */
template <typename PixelT>
std::shared_ptr<lsst::afw::image::MaskedImage<PixelT>> statisticsStack(
        std::vector<std::shared_ptr<lsst::afw::image::MaskedImage<PixelT>>>& images, Property flags,
        StatisticsControl const& sctrl, std::vector<lsst::afw::image::VariancePixel> const& wvector,
        image::MaskPixel clipped, std::vector<std::pair<image::MaskPixel, image::MaskPixel>> const& maskMap);

/**
 * A function to compute some statistics of a stack of Masked Images
 *
 * If none of the input images are valid for some pixel,
 * the afwMath::StatisticsControl::getNoGoodPixelsMask() bit(s) are set.
 *
 * Delegates to the more general version of statisticsStack taking a maskMap.
 */
template <typename PixelT>
std::shared_ptr<lsst::afw::image::MaskedImage<PixelT>> statisticsStack(
        std::vector<std::shared_ptr<lsst::afw::image::MaskedImage<PixelT>>>&
                images,                                        ///< MaskedImages to process
        Property flags,                                        ///< statistics requested
        StatisticsControl const& sctrl = StatisticsControl(),  ///< control structure
        std::vector<lsst::afw::image::VariancePixel> const& wvector =
                std::vector<lsst::afw::image::VariancePixel>(0),  ///< vector containing weights
        image::MaskPixel clipped = 0,  ///< bitmask to set if any input was clipped or masked
        image::MaskPixel excuse = 0    ///< bitmask to excuse from marking as clipped
);

/**
 * A function to compute some statistics of a stack of Masked Images
 *
 * @param[out] out      Output MaskedImage.
 * @param[in] images    MaskedImages to process.
 * @param[in] flags     Statistics requested.
 * @param[in] sctrl     Control structure.
 * @param[in] wvector   Vector of weights.
 * @param[in] clipped   Mask to set for pixels that were clipped (NOT rejected
 *                      due to masks).
 * @param[in] maskMap   Vector of pairs of mask pixel values; any pixel
 *                      on an input with any of the bits in .first will result
 *                      in all of the bits in .second being set on the
 *                      corresponding pixel on the output.
 *
 * If none of the input images are valid for some pixel,
 * the afwMath::StatisticsControl::getNoGoodPixelsMask() bit(s) are set.
 *
 * All the work is done in the function computeMaskedImageStack.
 */
template <typename PixelT>
void statisticsStack(lsst::afw::image::MaskedImage<PixelT>& out,
                     std::vector<std::shared_ptr<lsst::afw::image::MaskedImage<PixelT>>>& images,
                     Property flags, StatisticsControl const& sctrl,
                     std::vector<lsst::afw::image::VariancePixel> const& wvector, image::MaskPixel clipped,
                     std::vector<std::pair<image::MaskPixel, image::MaskPixel>> const& maskMap);

/**
 * @ brief compute statistical stack of MaskedImage.  Write to output image in-situ
 */
template <typename PixelT>
void statisticsStack(lsst::afw::image::MaskedImage<PixelT>& out,  ///< Output image
                     std::vector<std::shared_ptr<lsst::afw::image::MaskedImage<PixelT>>>&
                             images,                                        ///< MaskedImages to process
                     Property flags,                                        ///< statistics requested
                     StatisticsControl const& sctrl = StatisticsControl(),  ///< control structure
                     std::vector<lsst::afw::image::VariancePixel> const& wvector =
                             std::vector<lsst::afw::image::VariancePixel>(0),  ///< vector containing weights
                     image::MaskPixel clipped = 0,  ///< bitmask to set if any input was clipped or masked
                     image::MaskPixel excuse = 0    ///< bitmask to excuse from marking as clipped
);

/**
 * A function to compute some statistics of a stack of std::vectors
 */
template <typename PixelT>
std::vector<PixelT> statisticsStack(
        std::vector<std::vector<PixelT>>& vectors,  ///< Vectors to process
        Property flags,                                              ///< statistics requested
        StatisticsControl const& sctrl = StatisticsControl(),        ///< control structure
        std::vector<lsst::afw::image::VariancePixel> const& wvector =
                std::vector<lsst::afw::image::VariancePixel>(0)  ///< vector containing weights
);

/* ****************************************************************** *
 *
 * x,y stacks
 *
 * ******************************************************************* */

/**
 * A function to compute statistics on the rows or columns of an image
 */
template <typename PixelT>
std::shared_ptr<lsst::afw::image::MaskedImage<PixelT>> statisticsStack(
        lsst::afw::image::Image<PixelT> const& image, Property flags, char dimension,
        StatisticsControl const& sctrl = StatisticsControl());
/**
 * A function to compute statistics on the rows or columns of an image
 */
template <typename PixelT>
std::shared_ptr<lsst::afw::image::MaskedImage<PixelT>> statisticsStack(
        lsst::afw::image::MaskedImage<PixelT> const& image, Property flags, char dimension,
        StatisticsControl const& sctrl = StatisticsControl());
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
