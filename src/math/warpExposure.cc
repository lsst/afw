// -*- LSST-C++ -*- // fixed format comment for emacs

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
 * \file
 *
 * \ingroup afw
 *
 * \brief Support for warping an %image to a new Wcs.
 *
 * \author Nicole M. Silvestri and Russell Owen, University of Washington
 */

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "boost/cstdint.hpp" 
#include "boost/regex.hpp"

#include "lsst/pex/logging/Trace.h" 
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;

afwMath::Kernel::Ptr afwMath::LanczosWarpingKernel::clone() const {
    return afwMath::Kernel::Ptr(new afwMath::LanczosWarpingKernel(this->getOrder()));
}

/**
* @brief get the order of the kernel
*/
int afwMath::LanczosWarpingKernel::getOrder() const {
    return this->getWidth() / 2;
}

afwMath::Kernel::Ptr afwMath::BilinearWarpingKernel::clone() const {
    return afwMath::Kernel::Ptr(new afwMath::BilinearWarpingKernel());
}

/**
* \brief Solve bilinear equation; the only permitted arguments are 0 or 1
*
* \throw lsst::pex::exceptions::InvalidParameterException if argument is not 0 or 1
*/
afwMath::Kernel::Pixel afwMath::BilinearWarpingKernel::BilinearFunction1::operator() (
    double x
) const {
#if 0 && !defined(NDEBUG)
    if (x == 0.0) {
        return 1.0 - this->_params[0];
    } else if (x == 1.0) {
        return this->_params[0];
    } else {                            // the mere presence of this check slows the call by 3 times
        std::ostringstream errStream;
        errStream << "x = " << x << "; must be 0 or 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, errStream.str());
    }
#else
    if (x == 0.0) {
        return 1.0 - this->_params[0];
    } else {
        return this->_params[0];
    }
#endif
}

/**
 * \brief Return string representation.
 */
std::string afwMath::BilinearWarpingKernel::BilinearFunction1::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << "_BilinearFunction1: ";
    os << Function1<Kernel::Pixel>::toString(prefix);
    return os.str();
}

afwMath::Kernel::Ptr afwMath::NearestWarpingKernel::clone() const {
    return afwMath::Kernel::Ptr(new afwMath::NearestWarpingKernel());
}

/**
* \brief Solve nearest neighbor equation; the only permitted arguments are 0 or 1
*
* \throw lsst::pex::exceptions::InvalidParameterException if argument is not 0 or 1
*/
afwMath::Kernel::Pixel afwMath::NearestWarpingKernel::NearestFunction1::operator() (
    double x
) const {
    if (x == 0.0) {
        return this->_params[0] < 0.5 ? 1.0 : 0.0;
    } else if (x == 1.0) {
        return this->_params[0] < 0.5 ? 0.0 : 1.0;
    } else {
        std::ostringstream errStream;
        errStream << "x = " << x << "; must be 0 or 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, errStream.str());
    }
}

/**
 * \brief Return string representation.
 */
std::string afwMath::NearestWarpingKernel::NearestFunction1::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << "_NearestFunction1: ";
    os << Function1<Kernel::Pixel>::toString(prefix);
    return os.str();
}

/**
 * \brief Return a warping kernel given its name.
 *
 * Intended for use with warpImage() and warpExposure().
 *
 * Allowed names are:
 * - bilinear: return a BilinearWarpingKernel
 * - lanczos#: return a LanczosWarpingKernel of order #, e.g. lanczos4
 * - nearest: return a NearestWarpingKernel
 */
boost::shared_ptr<afwMath::SeparableKernel> afwMath::makeWarpingKernel(std::string name) {
    typedef boost::shared_ptr<afwMath::SeparableKernel> KernelPtr;
    boost::cmatch matches;
    static const boost::regex LanczosRE("lanczos(\\d+)");
    if (name == "bilinear") {
        return KernelPtr(new BilinearWarpingKernel());
    } else if (boost::regex_match(name.c_str(), matches, LanczosRE)) {
        std::string orderStr(matches[1].first, matches[1].second);
        int order;
        std::istringstream(orderStr) >> order;
        return KernelPtr(new LanczosWarpingKernel(order));
    } else if (name == "nearest") {
        return KernelPtr(new NearestWarpingKernel());
    } else {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
            "unknown warping kernel name: \"" + name + "\"");
    }
}

/**
 * \brief Convenience wrapper around warpImage()
 */
template<typename DestExposureT, typename SrcExposureT>
int afwMath::warpExposure(
    DestExposureT &destExposure,        ///< remapped exposure
    SrcExposureT const &srcExposure,    ///< source exposure
    SeparableKernel &warpingKernel,     ///< warping kernel; determines warping algorithm
    int const interpLength              ///< Distance over which WCS can be linearily interpolated    
    )
{
    if (!destExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "destExposure has no Wcs");
    }
    if (!srcExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "srcExposure has no Wcs");
    }
    typename DestExposureT::MaskedImageT mi = destExposure.getMaskedImage();
    return warpImage(mi, *destExposure.getWcs(),
                     srcExposure.getMaskedImage(), *srcExposure.getWcs(), warpingKernel, interpLength);
}


/************************************************************************************************************/
namespace {
    inline std::pair<afwGeom::Point2D, float>
    getSrcPos(afwGeom::Point2D destPosXY,
              afwImage::Wcs const &destWcs,   ///< WCS of remapped %image
              afwImage::Wcs const &srcWcs,    ///< WCS of source %image
              std::vector<afwGeom::Point2D>::iterator const prevSrcPosXY)
    {
        /*
         * These two versions are equivalent, but the second is faster as it doesn't need to build a Coord
         */
#if 0
        afwGeom::Point2D srcPosXY = srcWcs.skyToPixel(destWcs.pixelToSky(destPosXY));
#else
        double const x = destPosXY[0];
        double const y = destPosXY[1];
        afwGeom::Point2D sky = destWcs.pixelToSky(x, y, true);
        afwGeom::Point2D srcPosXY = srcWcs.skyToPixel(sky[0], sky[1]);
#endif
        // Correct intensity due to relative pixel spatial scale and kernel sum.
        // The area computation is for a parallellogram.
        afwGeom::Point2D dSrcA = srcPosXY - afwGeom::Extent<double>(prevSrcPosXY[-1]);
        afwGeom::Point2D dSrcB = srcPosXY - afwGeom::Extent<double>(prevSrcPosXY[0]);
        
        return std::make_pair(srcPosXY,
                              std::abs(dSrcA.getX()*dSrcB.getY() - dSrcA.getY()*dSrcB.getX()));
    }
}

/**
 * \brief Remap an Image or MaskedImage to a new Wcs. See also convenience function
 * warpExposure() to warp an Exposure.
 *
 * Edge pixels of destImage are set to the value returned by edgePixel().
 * These are pixels whose data comes from too near the edge of srcImage, or misses srcImage entirely.
 *
 * \return the number of valid pixels in destImage (those that are not edge pixels).
 *
 * \b Warping \b Kernels:
 *
 * This function requires a warping kernel to perform the interpolation.
 * Available options include:
 * - BilinearWarpingKernel
 * - LanczosWarpingKernel
 * - NearestWarpingKernel (nearest neighbor)
 *
 * makeWarpingKernel() is a handy factory function for constructing a warping kernel given its name.
 *
 * A warping kernel is a subclass of SeparableKernel with the following properties:
 * - It has two parameters: fractional x and fractional y position on the source %image.
 *   The fractional position for each axis is in the range [0, 1):
 *   - 0 if the position on the source along that axis is on the center of the pixel.
 *   - 0.999... if the position on the source along that axis is almost on the center of the next pixel.
 * - It almost always has even width and height (which is unusual for a kernel) and a center index of
 *   (width/2, /height/2). This is because the kernel is used to map source positions that range from
 *   centered on on pixel (width/2, height/2) to nearly centered on pixel (width/2 + 1, height/2 + 1).
 *
 * \b Algorithm:
 *
 * For each integer pixel position in the remapped Exposure:
 * - The associated sky coordinates are determined using the remapped WCS.
 * - The associated pixel position on srcImage is determined using the source WCS.
 * - A remapping kernel is computed based on the fractional part of the pixel position on srcImage
 * - The remapping kernel is applied to srcImage at the integer portion of the pixel position
 *   to compute the remapped pixel value
 * - The flux-conserving factor is determined from the source and new WCS.
 *   and is applied to the remapped pixel
 *
 * The scaling of intensity for relative area of source and destination uses two approximations:
 * - The area of the sky marked out by a pixel on the destination %image
 *   corresponds to a parallellogram on the source %image.
 * - The area varies slowly enough across the %image that we can get away with computing
 *   the source area shifted by half a pixel up and to the left of the true area.
 *
 * \throw lsst::pex::exceptions::InvalidParameterException if destImage is srcImage
 *
 * \todo Should support an additional color-based position correction in the remapping (differential chromatic
 *   refraction). This can be done either object-by-object or pixel-by-pixel.
 *
 * \todo Need to deal with oversampling and/or weight maps. If done we can use faster kernels than sinc.
 */
template<typename DestImageT, typename SrcImageT>
int afwMath::warpImage(
    DestImageT &destImage,              ///< remapped %image
    afwImage::Wcs const &destWcs,       ///< WCS of remapped %image
    SrcImageT const &srcImage,          ///< source %image
    afwImage::Wcs const &srcWcs,        ///< WCS of source %image
    SeparableKernel &warpingKernel,     ///< warping kernel; determines warping algorithm
    int const interpLength              ///< Distance over which WCS can be linearily interpolated
    )
{
    if (afwMath::details::isSameObject(destImage, srcImage)) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
            "destImage is srcImage; cannot warp in place");
    }
    int numGoodPixels = 0;

    typedef afwImage::Image<afwMath::Kernel::Pixel> KernelImageT;
    
    // Compute borders; use to prevent applying kernel outside of srcImage
    const int kernelWidth = warpingKernel.getWidth();
    const int kernelHeight = warpingKernel.getHeight();
    const int kernelCtrX = warpingKernel.getCtrX();
    const int kernelCtrY = warpingKernel.getCtrY();

    // Get the source MaskedImage and a pixel accessor to it.
    const int srcWidth = srcImage.getWidth();
    const int srcHeight = srcImage.getHeight();

    pexLog::TTrace<3>("lsst.afw.math.warp", "source image width=%d; height=%d", srcWidth, srcHeight);

    const int destWidth = destImage.getWidth();
    const int destHeight = destImage.getHeight();
    pexLog::TTrace<3>("lsst.afw.math.warp", "remap image width=%d; height=%d", destWidth, destHeight);

    const typename DestImageT::SinglePixel edgePixel = afwMath::edgePixel<DestImageT>(
        typename afwImage::detail::image_traits<DestImageT>::image_category()
    );
    
    std::vector<double> kernelXList(kernelWidth);
    std::vector<double> kernelYList(kernelHeight);

    // Set each pixel of destExposure's MaskedImage
    pexLog::TTrace<4>("lsst.afw.math.warp", "Remapping masked image");
    
    std::vector<afwGeom::Point2D> _srcPosXY(1 + destWidth);
    std::vector<afwGeom::Point2D>::iterator srcPosXY = _srcPosXY.begin() + 1;
    // compute source position X,Y corresponding to row -1 of the destination image;
    // this is used for computing relative pixel scale
    for (int x = -1; x != destWidth; ++x) {
        afwGeom::Point2D destPosXY = afwGeom::Point2D(destImage.indexToPosition(x, afwImage::X),
                                                         destImage.indexToPosition(-1, afwImage::Y));
        srcPosXY[x] = srcWcs.skyToPixel(destWcs.pixelToSky(destPosXY));
    }
    //
    // We overallocate a pixel here, and make prevSrcPosXY point to second element (which will be pixel [0])
    // so that prevSrcPosXY[-1] is valid
    //
    std::vector<afwGeom::Point2D> _prevSrcPosXY(1 + destWidth); // previous row's srcPosXY vector
    std::vector<afwGeom::Point2D>::iterator prevSrcPosXY = _prevSrcPosXY.begin() + 1;
    std::vector<float> _relativeArea(1 + destWidth); // relative dest and src area for each pixel
    std::vector<float>::iterator relativeArea = _relativeArea.begin() + 1;
    
    afwGeom::Point2D oneSrcPosXY;
    for (int y = 0; y < destHeight; ++y) {
        //
        // Set prevSrcPosXY from last row's srcPosXY. Note that we overallocated a pixel,
        // so it's safe to set the [-1] element
        //
        std::copy(srcPosXY - 1, srcPosXY + destWidth, prevSrcPosXY - 1);
        //
        // Calculate the transformation for the pixel just to the left of this row
        //
        afwGeom::Point2D destPosXY = afwGeom::Point2D(destImage.indexToPosition(-1, afwImage::X),
                                                         destImage.indexToPosition(y, afwImage::Y));
        {
            std::pair<afwGeom::Point2D, float> res = getSrcPos(destPosXY, destWcs, srcWcs,
                                                               prevSrcPosXY);
            srcPosXY[-1] = res.first;
            relativeArea[-1] = res.second;
        }
        //
        // Compute the transformations for this row
        //
        // Rather than calculate the transformation for each pixel, we'll estimate it every interpLength
        // pixels
        //
#if 1
        if (interpLength < 1) {
            for (int x = 0; x < destWidth; ++x) {
                // compute sky position associated with this pixel of remapped MaskedImage
                destPosXY[0] = destImage.indexToPosition(x, afwImage::X);
                std::pair<afwGeom::Point2D, float> res =
                    getSrcPos(destPosXY, destWcs, srcWcs, prevSrcPosXY + x);
                srcPosXY[x] = res.first;
                relativeArea[x] = res.second;
            }
        } else {
            for (int x = 0; x < destWidth + interpLength; x += interpLength) {
                int interval = interpLength;
                int xend = x + interval - 1;
                if (xend >= destWidth) {
                    xend = destWidth - 1;
                    interval = xend - x + 1;
                }
                // compute sky position associated with [xend] pixel of remapped MaskedImage
                destPosXY[0] = destImage.indexToPosition(xend, afwImage::X);

                std::pair<afwGeom::Point2D, float> res = getSrcPos(destPosXY, destWcs, srcWcs,
                                                                   prevSrcPosXY + xend);
                srcPosXY[xend] = res.first;
                relativeArea[xend] = res.second;

                for (int i = 0; i < interval - 1; ++i) {
                    for (int j = 0; j != 2; ++j) {
                        srcPosXY[x + i].coeffRef(j) = srcPosXY[x - 1].coeffRef(j) +
                            (i + 1)*(srcPosXY[xend].coeffRef(j) - srcPosXY[x - 1].coeffRef(j))/interval;
                    }

                    relativeArea[x + i] = relativeArea[x - 1] +
                        (i + 1)*(relativeArea[xend] - relativeArea[x - 1])/interval;
                }
            }
        }
#else
        for (int x = 0; x < destWidth; ++x) {
            // compute sky position associated with this pixel of remapped MaskedImage
            destPosXY[0] = destImage.indexToPosition(x, afwImage::X);

            // Compute associated pixel position on source MaskedImage
            srcPosXY[x] = srcWcs.skyToPixel(destWcs.pixelToSky(destPosXY));
            {
                // Correct intensity due to relative pixel spatial scale and kernel sum.
                // The area computation is for a parallellogram.
                oneSrcPosXY = srcPosXY[x];
                afwGeom::Point2D dSrcA = oneSrcPosXY - afwGeom::Extent<double>(prevSrcPosXY[x - 1]);
                afwGeom::Point2D dSrcB = oneSrcPosXY - afwGeom::Extent<double>(prevSrcPosXY[x    ]);
                relativeArea[x] = std::abs(dSrcA.getX()*dSrcB.getY() - dSrcA.getY()*dSrcB.getX());
            }
        }
#endif
        
        typename DestImageT::x_iterator destXIter = destImage.row_begin(y);
        for (int x = 0; x < destWidth; ++x, ++destXIter) {
            oneSrcPosXY = srcPosXY[x];  // pixel position on source

            // Compute associated source pixel index as integer and nonnegative fractional parts;
            // the latter is used to compute the remapping kernel.
            std::pair<int, double> srcIndFracX = srcImage.positionToIndex(oneSrcPosXY[0], afwImage::X);
            std::pair<int, double> srcIndFracY = srcImage.positionToIndex(oneSrcPosXY[1], afwImage::Y);
            if (srcIndFracX.second < 0) {
                ++srcIndFracX.second;
                --srcIndFracX.first;
            }
            if (srcIndFracY.second < 0) {
                ++srcIndFracY.second;
                --srcIndFracY.first;
            }

            // Offset source pixel index from kernel center to kernel corner (0, 0)
            // so we can convolveAtAPoint the pixels that overlap between source and kernel
            srcIndFracX.first -= kernelCtrX;
            srcIndFracY.first -= kernelCtrY;
          
            // If location is too near the edge of the source, or off the source, mark the dest as edge
            if ((srcIndFracX.first < 0) || (srcIndFracX.first + kernelWidth > srcWidth) ||
                (srcIndFracY.first < 0) || (srcIndFracY.first + kernelHeight > srcHeight)) {
                // skip this pixel
                *destXIter = edgePixel;
            } else {
                ++numGoodPixels;
                    
                // Compute warped pixel
                std::pair<double, double> srcFracInd(srcIndFracX.second, srcIndFracY.second);
                warpingKernel.setKernelParameters(srcFracInd);
                double kSum = warpingKernel.computeVectors(kernelXList, kernelYList, false);

                typename SrcImageT::const_xy_locator srcLoc =
                    srcImage.xy_at(srcIndFracX.first, srcIndFracY.first);
                
                *destXIter = afwMath::convolveAtAPoint<DestImageT,SrcImageT>(srcLoc, kernelXList, kernelYList);
                *destXIter *= relativeArea[x]/kSum;
            }
        } // dest x pixels
    } // dest y pixels

    return numGoodPixels;
}


//
// Explicit instantiations
//
// may need to omit default params for EXPOSURE -- original code did that and it worked
#define EXPOSURE(PIXTYPE) afwImage::Exposure<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define NL /* */

#define INSTANTIATE(DESTIMAGEPIXELT, SRCIMAGEPIXELT) \
    template int afwMath::warpImage( \
        IMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        SeparableKernel &warpingKernel, int const interpLength); NL    \
    template int afwMath::warpImage( \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        SeparableKernel &warpingKernel, int const interpLength); NL    \
    template int afwMath::warpExposure( \
        EXPOSURE(DESTIMAGEPIXELT) &destExposure, \
        EXPOSURE(SRCIMAGEPIXELT) const &srcExposure, \
        SeparableKernel &warpingKernel, int const interpLength);

INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, boost::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, boost::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(boost::uint16_t, boost::uint16_t)
