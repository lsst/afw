// -*- LSST-C++ -*- // fixed format comment for emacs
/**
 * \file
 *
 * \ingroup afw
 *
 * \brief Implementation of the templated utility function, warpExposure, for
 * Astrometric Image Remapping for LSST.  Declared in warpExposure.h.
 *
 * \todo: Figure out a better EDGE pixel; max() is not so good
 * because any subsequent operation will then overflow. inf avoids that problem,
 * but doesn't work for int-like images.
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
 * \brief convenience wrapper around warpImage
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
 * \brief Remap an image or masked image to a new WCS.
 *
 * For pixels in destImage that cannot be computed because their data comes from pixels that are too close
 * to (or off of) the edge of srcImage:
 * * The image and variance are set to 0
 * * The mask is set to the EDGE bit (if found, else 0).
 *
 * \return the number of valid pixels in destImage (those that are not off the edge).
 *
 * Algorithm:
 *
 * For each integer pixel position in the remapped Exposure:
 * * The associated sky coordinates are determined using the remapped WCS.
 * * The associated pixel position on srcImage is determined using the source WCS.
 * * A remapping kernel is computed based on the fractional part of the pixel position on srcImage
 * * The remapping kernel is applied to srcImage at the integer portion of the pixel position
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
 * \throw lsst::pex::exceptions::InvalidParameterException if destImage is srcImage
 *
 * \todo Should support an additional color-based position correction in the remapping (differential chromatic
 *   refraction). This can be done either object-by-object or pixel-by-pixel.
 *
 * \todo Need to deal with oversampling and/or weight maps. If done we can use faster kernels than sinc.
 */
template<typename DestImageT, typename SrcImageT>
int afwMath::warpImage(
    DestImageT &destImage,       ///< remapped image
    afwImage::Wcs const &destWcs,   ///< WCS of remapped image
    SrcImageT const &srcImage,   ///< source image
    afwImage::Wcs const &srcWcs,    ///< WCS of source image
    SeparableKernel &warpingKernel  ///< warping kernel; determines warping algorithm
    )
{
    if (isSameObject(destImage, srcImage)) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "destImage is srcImage; cannot warp in place");
    }
    int numGoodPixels = 0;

    typedef afwImage::Image<afwMath::Kernel::PixelT> KernelImageT;
    
    // Compute borders; use to prevent applying kernel outside of srcImage
    const int kernelWidth = warpingKernel.getWidth();
    const int kernelHeight = warpingKernel.getHeight();
    const int kernelCtrX = warpingKernel.getCtrX();
    const int kernelCtrY = warpingKernel.getCtrY();

    // Get the source MaskedImage and a pixel accessor to it.
    const int srcWidth = srcImage.getWidth();
    const int srcHeight = srcImage.getHeight();

    lsst::pex::logging::TTrace<3>("lsst.afw.math.warp",
        "source image width=%d; height=%d", srcWidth, srcHeight);

    const int destWidth = destImage.getWidth();
    const int destHeight = destImage.getHeight();
    lsst::pex::logging::TTrace<3>("lsst.afw.math.warp",
        "remap image width=%d; height=%d", destWidth, destHeight);

    const typename DestImageT::SinglePixel edgePixel = afwMath::edgePixel<DestImageT>(
        typename lsst::afw::image::detail::image_traits<DestImageT>::image_category()
    );
    
    std::vector<double> kernelXList(kernelWidth);
    std::vector<double> kernelYList(kernelHeight);

    // Set each pixel of destExposure's MaskedImage
    lsst::pex::logging::TTrace<4>("lsst.afw.math.warp", "Remapping masked image");
    
    // compute source position X,Y corresponding to row -1 of the destination image;
    // this is used for computing relative pixel scale
    std::vector<afwImage::PointD> prevRowSrcPosXY(destWidth+1);
    for (int destIndX = 0; destIndX < destWidth; ++destIndX) {
        afwImage::PointD destPosXY(
            afwImage::indexToPosition(destIndX),
            afwImage::indexToPosition(-1));
        afwImage::PointD srcPosXY = srcWcs.raDecToXY(destWcs.xyToRaDec(destPosXY));
        prevRowSrcPosXY[destIndX] = srcPosXY;
    }
    for (int destIndY = 0; destIndY < destHeight; ++destIndY) {
        afwImage::PointD destPosXY(
            afwImage::indexToPosition(-1),
            afwImage::indexToPosition(destIndY));
        afwImage::PointD prevSrcPosXY = srcWcs.raDecToXY(destWcs.xyToRaDec(destPosXY));
        afwImage::PointD srcPosXY;
        typename DestImageT::x_iterator destXIter = destImage.row_begin(destIndY);
        for (int destIndX = 0; destIndX < destWidth; ++destIndX, ++destXIter) {
            // compute sky position associated with this pixel of remapped MaskedImage
            destPosXY[0] = afwImage::indexToPosition(destIndX);

            // Compute associated pixel position on source MaskedImage
            srcPosXY = srcWcs.raDecToXY(destWcs.xyToRaDec(destPosXY));

            // Compute associated source pixel index and break it into integer and fractional
            // parts; the latter is used to compute the remapping kernel.
            // To convolve at source pixel (x, y) point source accessor to (x - kernelCtrX, y - kernelCtrY)
            // because the accessor must point to kernel pixel (0, 0), not the center of the kernel.
            std::vector<double> srcFracInd(2);
            int srcIndX = afwImage::positionToIndex(srcFracInd[0], srcPosXY[0]) - kernelCtrX;
            int srcIndY = afwImage::positionToIndex(srcFracInd[1], srcPosXY[1]) - kernelCtrY;
            if (srcFracInd[0] < 0) {
                ++srcFracInd[0];
                --srcIndX;
            }
            if (srcFracInd[1] < 0) {
                ++srcFracInd[1];
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
                *destXIter = afwMath::convolveAtAPoint<DestImageT, SrcImageT>(srcLoc, kernelXList, kernelYList);
    
                // Correct intensity due to relative pixel spatial scale and kernel sum.
                // The area computation is for a parallellogram.
                afwImage::PointD dSrcA = srcPosXY - prevSrcPosXY;
                afwImage::PointD dSrcB = srcPosXY - prevRowSrcPosXY[destIndX];
                double multFac = std::abs((dSrcA.getX() * dSrcB.getY()) - (dSrcA.getY() * dSrcB.getX())) / kSum;
                *destXIter *= multFac;
//                destXIter.image() *= static_cast<typename DestImageT::Image::SinglePixel>(multFac);
//                destXIter.variance() *= static_cast<typename DestImageT::Variance::SinglePixel>(multFac * multFac);
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
