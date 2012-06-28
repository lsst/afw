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
#include "lsst/afw/math/detail/computeOneWarpedPixel.h"

namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;

/**
 * @brief Compute one warped pixel
 *
 * This is the Image specialization; it ignores the mask kernel.
 */
template<typename ToPixelT, typename FromPixelT>
void afwMath::detail::computeOneWarpedPixel(
    typename afwImage::Image<ToPixelT>::x_iterator &destXIter,
    WarpingKernelInfo &kernelInfo,
    afwImage::Image<FromPixelT> const &srcImage,
    afwGeom::Point2D const &srcPos,
    double relativeArea,
    typename afwImage::Image<ToPixelT>::SinglePixel const &padValue
) {
    // Compute associated source pixel index as integer and nonnegative fractional parts;
    // the latter is used to compute the remapping kernel.
    std::pair<int, double> srcIndFracX = srcImage.positionToIndex(srcPos[0], afwImage::X);
    std::pair<int, double> srcIndFracY = srcImage.positionToIndex(srcPos[1], afwImage::Y);
    if (srcIndFracX.second < 0) {
        ++srcIndFracX.second;
        --srcIndFracX.first;
    }
    if (srcIndFracY.second < 0) {
        ++srcIndFracY.second;
        --srcIndFracY.first;
    }

    if (srcGoodBBox.contains(afwGeom::Point2I(srcIndFracX.first, srcIndFracY.first))) {
         ++numGoodPixels;

        // Offset source pixel index from kernel center to kernel corner (0, 0)
        // so we can convolveAtAPoint the pixels that overlap between source and kernel
        afwGeom::Point2I kernelCtr = kernelInfo.getKernelCtr();
        int srcStartX = srcIndFracX.first - kernelCtr[0];
        int srcStartY = srcIndFracY.first - kernelCtr[1];

        // Compute warped pixel
        double kSum = kernelInfo.setFracIndex(srcIndFracX.second, srcIndFracY.second);

        typename SrcImageT::const_xy_locator srcLoc = srcImage.xy_at(srcStartX, srcStartY);

        *destXIter = afwMath::convolveAtAPoint<DestImageT,SrcImageT>(
            srcLoc, kernelInfo.getXList(), kernelInfo.getYList());
        *destXIter *= relativeArea/kSum;
    } else {
       // Edge pixel pixel
        *destXIter = padValue;
    }
}

/**
 * @brief Compute one warped pixel
 *
 * This is the MaskedImage specialization. It uses the mask kernel, if present, to compute the mask pixel.
 */
template<typename ToPixelT, typename FromPixelT>
void afwMath::detail::computeOneWarpedPixel(
    typename afwImage::MaskedImage<ToPixelT>::x_iterator &destXIter,
    WarpingKernelInfo &kernelInfo,
    afwImage::MaskedImage<FromPixelT> const &srcImage,
    afwGeom::Point2D const &srcPos,
    double relativeArea,
    typename afwImage::MaskedImage<ToPixelT>::SinglePixel const &padValue
) {
    // Compute associated source pixel index as integer and nonnegative fractional parts;
    // the latter is used to compute the remapping kernel.
    std::pair<int, double> srcIndFracX = srcImage.positionToIndex(srcPos[0], afwImage::X);
    std::pair<int, double> srcIndFracY = srcImage.positionToIndex(srcPos[1], afwImage::Y);
    if (srcIndFracX.second < 0) {
        ++srcIndFracX.second;
        --srcIndFracX.first;
    }
    if (srcIndFracY.second < 0) {
        ++srcIndFracY.second;
        --srcIndFracY.first;
    }

    if (srcGoodBBox.contains(afwGeom::Point2I(srcIndFracX.first, srcIndFracY.first))) {
         ++numGoodPixels;

        // Offset source pixel index from kernel center to kernel corner (0, 0)
        // so we can convolveAtAPoint the pixels that overlap between source and kernel
        afwGeom::Point2I kernelCtr = kernelInfo.getKernelCtr();
        int srcStartX = srcIndFracX.first - kernelCtr[0];
        int srcStartY = srcIndFracY.first - kernelCtr[1];

        // Compute warped pixel
        double kSum = kernelInfo.setFracIndex(srcIndFracX.second, srcIndFracY.second);

        typename SrcImageT::const_xy_locator srcLoc = srcImage.xy_at(srcStartX, srcStartY);

        *destXIter = afwMath::convolveAtAPoint<DestImageT,SrcImageT>(
            srcLoc, kernelInfo.getXList(), kernelInfo.getYList());
        *destXIter *= relativeArea/kSum;
        
        if (kernelInfo.hasMaskKernel()) {
            // compute mask value based on the mask kernel (replacing the value computed above)
            int maskStartX = srcIndFracX.first - kernelCtr[0];
            int maskStartY = srcIndFracY.first - kernelCtr[1];

            typename SrcImageT::Mask::const_xy_locator srcMaskLoc = srcImage.xy_at(maskStartX, maskStartY);
    
            typedef typename std::vector<lsst::afw::math::Kernel::Pixel>::const_iterator k_iter;
        
            afwImage::MaskT destMaskValue = 0;
            for (k_iter kernelYIter = kernelInfo.getMaskYList().begin(), yEnd = kernelInfo.getMaskYList().end();
                 kernelYIter != yEnd; ++kernelYIter) {
        
                afwImage::MaskT destMaskValueY = 0;
                for (k_iter kernelXIter = kernelInfo.getMaskXList().begin(), xEnd = kernelInfo.getMaskXList().end();
                     kernelXIter != xEnd; ++kernelXIter, ++srcMaskLoc.x()) {
                    typename lsst::afw::math::Kernel::Pixel const kValX = *kernelXIter;
                    if (kValX != 0) {
                        destMaskValueY |= *srcMaskLoc;
                    }
                }
        
                double const kValY = *kernelYIter;
                if (kValY != 0) {
                    destMaskValue |= destMaskValueY;
                }
        
                srcMaskLoc += lsst::afw::image::detail::difference_type(-kernelInfo.getMaskXList().size(), 1);
            }
    
    
            *destXIter::GET_MASK_HOW??? = destMaskValue;
        }
    } else {
       // Edge pixel pixel
        *destXIter = padValue;
    }
}
