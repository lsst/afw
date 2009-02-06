// -*- LSST-C++ -*- // fixed format comment for emacs
/**
 * @file
 *
 * @ingroup afw
 *
 * @brief Implementation of the templated utility function, warpExposure, for
 * Astrometric Image Remapping for LSST.  Declared in warpExposure.h.
 *
 * @author Nicole M. Silvestri and Russell Owen, University of Washington
 */

#include <string>
#include <vector>
#include <cmath>

#include <boost/cstdint.hpp> 
#include <boost/format.hpp> 

#include "lsst/pex/logging/Trace.h" 
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

/**
* @brief Solve bilinear equation; the only permitted arguments are 0 or 1
*
* @throw lsst::pex::exceptions::InvalidParameterException if argument is not 0 or 1
*/
afwMath::Kernel::PixelT afwMath::BilinearWarpingKernel::BilinearFunction1::operator() (
    double x
) const {
    if (x == 0.0) {
        return 1.0 - this->_params[0];
    } else if (x == 1.0) {
        return this->_params[0];
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "x must be 0 or 1");
    }
}            

/**
* @brief Return string representation.
*/
std::string afwMath::BilinearWarpingKernel::BilinearFunction1::toString(void) const {
    std::ostringstream os;
    os << "_BilinearFunction1: ";
    os << Function1<Kernel::PixelT>::toString();
    return os.str();
}


/**
 * @brief Remap an Exposure to a new WCS.
 *
 * For pixels in destExposure that cannot be computed because their data comes from pixels that are too close
 * to (or off of) the edge of srcExposure.
 * * The image and variance are set to 0
 * * The mask is set to the EDGE bit (if found, else 0).
 *
 * @return the number valid pixels in destExposure (thost that are not off the edge).
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
 * A warping kernel has the following properties:
 * - Has two parameters: fractional x and fractional y position on the source image.
 *   The fractional position for each axis has value >= 0 and < 1:
 *   0 if the center of the source along that axis is on the center of the pixel
 *   0.999... if the center of the source along that axis is almost on the center of the next pixel
 * - Almost always has even width and height (unusual for a kernel) and a center index = width/height/2.
 *   This is because the kernel is used to map from a range of pixel positions from
 *   centered on on (width/2, height/2) to nearly centered on (1 + width/2, 1 + height/2).
 *
 * TODO 20071129 Nicole M. Silvestri; By DC3:
 * * Need to synchronize warpExposure to the UML model robustness/sequence diagrams.
 *   Remove from the Exposure Class in the diagrams.
 *
 * * Should support an additional color-based position correction in the remapping (differential chromatic
 *   refraction). This can be done either object-by-object or pixel-by-pixel.
 *
 * * Need to deal with oversampling and/or weight maps. If done we can use faster kernels than sinc.
 */
template<typename DestExposureT, typename SrcExposureT>
int afwMath::warpExposure(
    DestExposureT &destExposure,        ///< remapped exposure
    SrcExposureT const &srcExposure,    ///< source exposure
    SeparableKernel &warpingKernel      ///< warping kernel; determines warping algorithm
    )
{
    int numGoodPixels = 0;

    typedef typename DestExposureT::MaskedImage DestMaskedImageT;
    typedef typename SrcExposureT::MaskedImage SrcMaskedImageT;
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
    typename afwImage::Wcs::Ptr srcWcsPtr = srcExposure.getWcs();
    lsst::pex::logging::Trace("lsst.afw.math", 3,
        boost::format("source image width=%d; height=%d") % srcWidth % srcHeight);

    // Get the remapped MaskedImage and the remapped wcs.
    DestMaskedImageT destMI = destExposure.getMaskedImage();
    typename afwImage::Wcs::Ptr destWcsPtr = destExposure.getWcs();
   
    // Conform mask plane names of remapped MaskedImage to match source
    destMI.getMask()->conformMaskPlanes(srcMI.getMask()->getMaskPlaneDict());
    
    // Make a pixel mask from the EDGE bit, if available (0 if not available)
    const typename DestMaskedImageT::Mask::SinglePixel edgePixelMask = srcMI.getMask()->getPlaneBitMask("EDGE");
    lsst::pex::logging::Trace("lsst.afw.math", 3, boost::format("edgePixelMask=0x%X") % edgePixelMask);
    
    const int destWidth = destMI.getWidth();
    const int destHeight = destMI.getHeight();
    lsst::pex::logging::Trace("lsst.afw.math", 3,
        boost::format("remap image width=%d; height=%d") % destWidth % destHeight);

    const typename DestMaskedImageT::SinglePixel edgePixel(0, edgePixelMask, 0);
    
    std::vector<double> kernelXList(kernelWidth);
    std::vector<double> kernelYList(kernelHeight);

    // Set each pixel of destExposure's MaskedImage
    lsst::pex::logging::Trace("lsst.afw.math", 4, "Remapping masked image");
    for (int destIndY = 0; destIndY < destHeight; ++destIndY) {
        afwImage::PointD destPosXY(0.0, afwImage::indexToPosition(destIndY));
        typename DestMaskedImageT::x_iterator destXIter = destMI.row_begin(destIndY);
        for (int destIndX = 0; destIndX < destWidth; ++destIndX, ++destXIter) {
            // compute sky position associated with this pixel of remapped MaskedImage
            destPosXY[0] = afwImage::indexToPosition(destIndX);

            afwImage::PointD raDec = destWcsPtr->xyToRaDec(destPosXY);            
            
            // Compute associated pixel position on source MaskedImage
            afwImage::PointD srcPosXY = srcWcsPtr->raDecToXY(raDec);

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
                continue;
            }
            
            ++numGoodPixels;

            // Compute warped pixel
            warpingKernel.setKernelParameters(srcFracInd);
            double kSum = warpingKernel.computeVectors(kernelXList, kernelYList, false);
            typename SrcMaskedImageT::const_xy_locator srcLoc = srcMI.xy_at(srcIndX, srcIndY);
            *destXIter = afwMath::convolveAtAPoint<DestMaskedImageT, SrcMaskedImageT>(srcLoc, kernelXList, kernelYList);

            // Correct intensity due to relative pixel spatial scale and kernel sum
            double multFac = srcWcsPtr->pixArea(srcPosXY) / (destWcsPtr->pixArea(destPosXY) * kSum);
            destXIter.image() *= static_cast<typename DestMaskedImageT::Image::SinglePixel>(multFac);
            destXIter.variance() *= static_cast<typename DestMaskedImageT::Variance::SinglePixel>(multFac * multFac);

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
