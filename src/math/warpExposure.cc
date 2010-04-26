// -*- LSST-C++ -*- // fixed format comment for emacs
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
#include "boost/format.hpp" 
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
    if (x == 0.0) {
        return 1.0 - this->_params[0];
    } else if (x == 1.0) {
        return this->_params[0];
    } else {
        std::ostringstream errStream;
        errStream << "x = " << x << "; must be 0 or 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, errStream.str());
    }
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
    const boost::regex LanczosRE("lanczos(\\d+)");
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
    SeparableKernel &warpingKernel      ///< warping kernel; determines warping algorithm
    )
{
    if (!destExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "destExposure has no Wcs");
    }
    if (!srcExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "srcExposure has no Wcs");
    }
    return warpImage(destExposure.getMaskedImage(), *destExposure.getWcs(),
        srcExposure.getMaskedImage(), *srcExposure.getWcs(), warpingKernel);
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
    DestImageT &destImage,       ///< remapped %image
    afwImage::Wcs const &destWcs,   ///< WCS of remapped %image
    SrcImageT const &srcImage,   ///< source %image
    afwImage::Wcs const &srcWcs,    ///< WCS of source %image
    SeparableKernel &warpingKernel  ///< warping kernel; determines warping algorithm
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
    
    // compute source position X,Y corresponding to row -1 of the destination image;
    // this is used for computing relative pixel scale
    std::vector<afwGeom::PointD> prevRowSrcPosXY(destWidth+1);
    for (int destIndX = 0; destIndX < destWidth; ++destIndX) {
        afwGeom::PointD destPosXY = afwGeom::makePointD(afwImage::indexToPosition(destIndX),
                                                        afwImage::indexToPosition(-1));
        afw::geom::PointD srcPosXY = srcWcs.skyToPixel(destWcs.pixelToSky(destPosXY));
        prevRowSrcPosXY[destIndX] = srcPosXY;
    }
    for (int destIndY = 0; destIndY < destHeight; ++destIndY) {
        afwGeom::PointD destPosXY = afwGeom::makePointD(afwImage::indexToPosition(-1),
                                                        afwImage::indexToPosition(destIndY));
        afw::geom::PointD prevSrcPosXY = srcWcs.skyToPixel(destWcs.pixelToSky(destPosXY));
        afw::geom::PointD srcPosXY;
        typename DestImageT::x_iterator destXIter = destImage.row_begin(destIndY);
        for (int destIndX = 0; destIndX < destWidth; ++destIndX, ++destXIter) {
            // compute sky position associated with this pixel of remapped MaskedImage
            destPosXY[0] = afwImage::indexToPosition(destIndX);

            // Compute associated pixel position on source MaskedImage
            srcPosXY = srcWcs.skyToPixel(destWcs.pixelToSky(destPosXY));

            // Compute associated source pixel index and break it into integer and fractional
            // parts; the latter is used to compute the remapping kernel.
            // To convolve at source pixel (x, y) point source accessor to (x - kernelCtrX, y - kernelCtrY)
            // because the accessor must point to kernel pixel (0, 0), not the center of the kernel.
            std::pair<double, double> srcFracInd;
            int srcIndX = afwImage::positionToIndex(srcFracInd.first,  srcPosXY[0]) - kernelCtrX;
            int srcIndY = afwImage::positionToIndex(srcFracInd.second, srcPosXY[1]) - kernelCtrY;
            if (srcFracInd.first < 0) {
                ++srcFracInd.first;
                --srcIndX;
            }
            if (srcFracInd.second < 0) {
                ++srcFracInd.second;
                --srcIndY;
            }
          
            // If location is too near the edge of the source, or off the source, mark the dest as edge
            if ((srcIndX < 0) || (srcIndX + kernelWidth > srcWidth) 
                || (srcIndY < 0) || (srcIndY + kernelHeight > srcHeight)) {
                // skip this pixel
                *destXIter = edgePixel;
            } else {
                ++numGoodPixels;
                    
                // Compute warped pixel
                warpingKernel.setKernelParameters(srcFracInd);
                double kSum = warpingKernel.computeVectors(kernelXList, kernelYList, false);

                typename SrcImageT::const_xy_locator srcLoc = srcImage.xy_at(srcIndX, srcIndY);
                *destXIter = afwMath::convolveAtAPoint<DestImageT, SrcImageT>(
                    srcLoc, kernelXList, kernelYList);
    
                // Correct intensity due to relative pixel spatial scale and kernel sum.
                // The area computation is for a parallellogram.
                afwGeom::PointD dSrcA = srcPosXY - afwGeom::Extent<double>(prevSrcPosXY);
                afwGeom::PointD dSrcB = srcPosXY - afwGeom::Extent<double>(prevRowSrcPosXY[destIndX]);
                double multFac = std::abs((dSrcA.getX() * dSrcB.getY())
                    - (dSrcA.getY() * dSrcB.getX())) / kSum;
                *destXIter *= multFac;
//                destXIter.image() *= static_cast<typename DestImageT::Image::SinglePixel>(multFac);
//                destXIter.variance() *=
//                    static_cast<typename DestImageT::Variance::SinglePixel>(multFac * multFac);
            }

            // Copy srcPosXY to prevRowSrcPosXY to use for computing area scaling for pixels in the next row
            // (we've finished with that value in prevRowSrcPosXY for this row)
            // and to prevSrcPosXY for computation the area scaling of the next pixel in this row
            prevRowSrcPosXY[destIndX] = srcPosXY;
            prevSrcPosXY = srcPosXY;

        } // dest x pixels
    } // dest y pixels
    return numGoodPixels;
} // warpExposure


//
// Explicit instantiations
//
// may need to omit default params for EXPOSURE -- original code did that and it worked
#define EXPOSURE(PIXTYPE) afwImage::Exposure<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define NL /* */

#define WarpFunctionsByType(DESTIMAGEPIXELT, SRCIMAGEPIXELT) \
    template int afwMath::warpImage( \
        IMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        SeparableKernel &warpingKernel); NL \
    template int afwMath::warpImage( \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        SeparableKernel &warpingKernel); NL \
    template int afwMath::warpExposure( \
        EXPOSURE(DESTIMAGEPIXELT) &destExposure, \
        EXPOSURE(SRCIMAGEPIXELT) const &srcExposure, \
        SeparableKernel &warpingKernel);

WarpFunctionsByType(float, boost::uint16_t)
WarpFunctionsByType(double, boost::uint16_t)
WarpFunctionsByType(float, int)
WarpFunctionsByType(double, int)
WarpFunctionsByType(float, float)
WarpFunctionsByType(double, float)
WarpFunctionsByType(double, double)
