// -*- LSST-C++ -*- // fixed format comment for emacs
/**
 * \file
 *
 * \ingroup afw
 *
 * \brief Implementation of the templated utility function, warpExposure, for
 * Astrometric Image Remapping for LSST.  Declared in warpExposure.h.
 *
 * \author Nicole M. Silvestri and Russell Owen, University of Washington
 */

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <boost/cstdint.hpp> 
#include <boost/format.hpp> 
#include <boost/regex.hpp>

#include "lsst/pex/logging/Trace.h" 
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace pexExcept = lsst::pex::exceptions;

namespace {
    template <typename A, typename B>
    bool isSameObject(A const& a, B const& b) {
        return false;
    }
    
    template <typename A>
    bool isSameObject(A const& a, A const& b) {
        return &a == &b;
    }
}

/**
* \brief Solve bilinear equation; the only permitted arguments are 0 or 1
*
* \throw lsst::pex::exceptions::InvalidParameterException if argument is not 0 or 1
*/
afwMath::Kernel::PixelT afwMath::BilinearWarpingKernel::BilinearFunction1::operator() (
    double x
) const {
    if (x == 0.0) {
        return 1.0 - this->_params[0];
    } else if (x == 1.0) {
        return this->_params[0];
    } else {
        std::ostringstream errStream;
        errStream << "x = " << x << "; must be 0 or 1";
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, errStream.str());
    }
}            

/**
 * \brief Return string representation.
 */
std::string afwMath::BilinearWarpingKernel::BilinearFunction1::toString(void) const {
    std::ostringstream os;
    os << "_BilinearFunction1: ";
    os << Function1<Kernel::PixelT>::toString();
    return os.str();
}

/**
 * \brief Return a warping kernel given its name
 *
 * Allowed names are:
 * * bilinear
 * * lanczosN where N is an integer, e.g. lanczos4
 */
boost::shared_ptr<lsst::afw::math::SeparableKernel> lsst::afw::math::makeWarpingKernel(std::string name) {
    typedef boost::shared_ptr<lsst::afw::math::SeparableKernel> KernelPtr;
    boost::cmatch matches;
    const boost::regex LanczosRE("lanczos(\\d+)");
    if (name == "bilinear") {
        return KernelPtr(new BilinearWarpingKernel());
    } else if (boost::regex_match(name.c_str(), matches, LanczosRE)) {
        std::string orderStr(matches[1].first, matches[1].second);
        int order;
        std::istringstream(orderStr) >> order;
        return KernelPtr(new LanczosWarpingKernel(order));
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
            "unknown warping kernel name: \"" + name + "\"");
    }
}

/**
 * \brief Remap an Exposure to a new WCS.
 *
 * For pixels in destExposure that cannot be computed because their data comes from pixels that are too close
 * to (or off of) the edge of srcExposure.
 * * The image and variance are set to 0
 * * The mask is set to the EDGE bit (if found, else 0).
 *
 * \return the number valid pixels in destExposure (thost that are not off the edge).
 *
 * Algorithm:
 *
 * For each integer pixel position in the remapped Exposure:
 * * The associated sky coordinates are determined using the remapped WCS.
 * * The associated pixel position on srcExposure is determined using the source WCS.
 * * A remapping kernel is computed based on the fractional part of the pixel position on srcExposure
 * * The remapping kernel is applied to srcExposure at the integer portion of the pixel position
 *   to compute the remapped pixel value
 * * The flux-conserving factor is determined from the source and new WCS.
 *   and is applied to the remapped pixel
 *
 * The scaling of intensity for relative area of source and destination uses two approximations:
 * - The area of the sky marked out by a pixel on the destination image
 *   corresponds to a parallellogram on the source image.
 * - The area varies slowly enough across the image that we can get away with computing
 *   the source area shifted by half a pixel up and to the left of the true area.
 *
 * A warping kernel has the following properties:
 * - Has two parameters: fractional x and fractional y position on the source image.
 *   The fractional position for each axis has value >= 0 and < 1:
 *   0 if the center of the source along that axis is on the center of the pixel
 *   0.999... if the center of the source along that axis is almost on the center of the next pixel
 * - Almost always has even width and height (unusual for a kernel) and a center index = width/height/2.
 *   This is because the kernel is used to map from a range of pixel positions from
 *   centered on on (width/2, height/2) to nearly centered on (1 + width/2, 1 + height/2).
 *
 * \throw lsst::pex::exceptions::InvalidParameterException if destExposure is srcExposure
 * \throw lsst::pex::exceptions::InvalidParameterException if destExposure or srcExposure has no Wcs
 *
 * \todo Should support an additional color-based position correction in the remapping (differential chromatic
 *   refraction). This can be done either object-by-object or pixel-by-pixel.
 *
 * \todo Need to deal with oversampling and/or weight maps. If done we can use faster kernels than sinc.
 */
template<typename DestExposureT, typename SrcExposureT>
int afwMath::warpExposure(
    DestExposureT &destExposure,        ///< remapped exposure
    SrcExposureT const &srcExposure,    ///< source exposure
    SeparableKernel &warpingKernel      ///< warping kernel; determines warping algorithm
    )
{
    if (isSameObject(destExposure, srcExposure)) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "destExposure is srcExposure; cannot warp in place");
    }
    int numGoodPixels = 0;

    typedef typename DestExposureT::MaskedImageT DestMaskedImageT;
    typedef typename SrcExposureT::MaskedImageT SrcMaskedImageT;
    typedef afwImage::Image<afwMath::Kernel::PixelT> KernelImageT;
    
    // Compute borders; use to prevent applying kernel outside of srcExposure
    const int kernelWidth = warpingKernel.getWidth();
    const int kernelHeight = warpingKernel.getHeight();
    const int kernelCtrX = warpingKernel.getCtrX();
    const int kernelCtrY = warpingKernel.getCtrY();

    // Get the source MaskedImage and a pixel accessor to it.
    SrcMaskedImageT srcMI = srcExposure.getMaskedImage();
    const int srcWidth = srcMI.getWidth();
    const int srcHeight = srcMI.getHeight();
    if (!srcExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "srcExposure has no Wcs");
    }
    typename afwImage::Wcs::Ptr srcWcsPtr = srcExposure.getWcs();

    lsst::pex::logging::TTrace<3>("lsst.afw.math.warp",
        "source image width=%d; height=%d", srcWidth, srcHeight);

    // Get the remapped MaskedImage and the remapped wcs.
    DestMaskedImageT destMI = destExposure.getMaskedImage();
    if (!destExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "destExposure has no Wcs");
    }
    typename afwImage::Wcs::Ptr destWcsPtr = destExposure.getWcs();
    
    // Make a pixel mask from the EDGE bit, if available (0 if not available)
    const typename DestMaskedImageT::Mask::SinglePixel edgePixelMask = srcMI.getMask()->getPlaneBitMask("EDGE");
    lsst::pex::logging::TTrace<3>("lsst.afw.math.warp", "edgePixelMask=0x%X", edgePixelMask);
    
    const int destWidth = destMI.getWidth();
    const int destHeight = destMI.getHeight();
    lsst::pex::logging::TTrace<3>("lsst.afw.math.warp",
        "remap image width=%d; height=%d", destWidth, destHeight);

    typedef typename DestMaskedImageT::Variance::Pixel VariancePixel;
    const typename DestMaskedImageT::SinglePixel edgePixel(
        0, edgePixelMask, std::numeric_limits<VariancePixel>::max());
    
    std::vector<double> kernelXList(kernelWidth);
    std::vector<double> kernelYList(kernelHeight);

    // Set each pixel of destExposure's MaskedImage
    lsst::pex::logging::TTrace<4>("lsst.afw.math.warp", "Remapping masked image");
    
    // compute source position X,Y corresponding to row -1 of the destination image;
    // this is used for computing relative pixel scale
    std::vector<afwImage::PointD> prevRowSrcPosXY(destWidth+1);
    for (int destIndX = 0; destIndX < destWidth; ++destIndX) {
        afwImage::PointD destPosXY(afwImage::indexToPosition(destIndX), afwImage::indexToPosition(-1));
        afwImage::PointD srcPosXY = srcWcsPtr->raDecToXY(destWcsPtr->xyToRaDec(destPosXY));
        prevRowSrcPosXY[destIndX] = srcPosXY;
    }
    for (int destIndY = 0; destIndY < destHeight; ++destIndY) {
        afwImage::PointD destPosXY(afwImage::indexToPosition(-1), afwImage::indexToPosition(destIndY));
        afwImage::PointD prevSrcPosXY = srcWcsPtr->raDecToXY(destWcsPtr->xyToRaDec(destPosXY));
        afwImage::PointD srcPosXY;
        typename DestMaskedImageT::x_iterator destXIter = destMI.row_begin(destIndY);
        for (int destIndX = 0; destIndX < destWidth; ++destIndX, ++destXIter) {
            // compute sky position associated with this pixel of remapped MaskedImage
            destPosXY[0] = afwImage::indexToPosition(destIndX);

            // Compute associated pixel position on source MaskedImage
            srcPosXY = srcWcsPtr->raDecToXY(destWcsPtr->xyToRaDec(destPosXY));

            // Compute associated source pixel index and break it into integer and fractional
            // parts; the latter is used to compute the remapping kernel.
            // To convolve at source pixel (x, y) point source accessor to (x - kernelCtrX, y - kernelCtrY)
            // because the accessor must point to kernel pixel (0, 0), not the center of the kernel.
            std::vector<double> srcFracInd(2);
            int srcIndX = afwImage::positionToIndex(srcFracInd[0], srcPosXY[0]) - kernelCtrX;
            int srcIndY = afwImage::positionToIndex(srcFracInd[1], srcPosXY[1]) - kernelCtrY;
          
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
                typename SrcMaskedImageT::const_xy_locator srcLoc = srcMI.xy_at(srcIndX, srcIndY);
                *destXIter = afwMath::convolveAtAPoint<DestMaskedImageT, SrcMaskedImageT>(srcLoc, kernelXList, kernelYList);
    
                // Correct intensity due to relative pixel spatial scale and kernel sum.
                // The area computation is for a parallellogram.
                afwImage::PointD dSrcA = srcPosXY - prevSrcPosXY;
                afwImage::PointD dSrcB = srcPosXY - prevRowSrcPosXY[destIndX];
                double multFac = std::abs((dSrcA.getX() * dSrcB.getY()) - (dSrcA.getY() * dSrcB.getX())) / kSum;
                destXIter.image() *= static_cast<typename DestMaskedImageT::Image::SinglePixel>(multFac);
                destXIter.variance() *= static_cast<typename DestMaskedImageT::Variance::SinglePixel>(multFac * multFac);
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


/************************************************************************************************************/
//
// Explicit instantiations
//
typedef float imagePixelType;

#define warpExposureFuncByType(DESTIMAGEPIXELT, SRCIMAGEPIXELT) \
    template int afwMath::warpExposure( \
        afwImage::Exposure<DESTIMAGEPIXELT> &destExposure, \
        afwImage::Exposure<SRCIMAGEPIXELT> const &srcExposure, \
        SeparableKernel &warpingKernel);


warpExposureFuncByType(float, boost::uint16_t)
warpExposureFuncByType(double, boost::uint16_t)
warpExposureFuncByType(float, int)
warpExposureFuncByType(double, int)
warpExposureFuncByType(float, float)
warpExposureFuncByType(double, float)
warpExposureFuncByType(double, double)
