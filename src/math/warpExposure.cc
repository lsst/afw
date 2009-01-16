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
 * * The mask bit EDGE is set, if present, else the mask pixel is set to 0
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
    DestExposureT &destExposure,      ///< remapped exposure
    SrcExposureT const &srcExposure, ///< source exposure
    SeparableKernel const &warpingKernel    ///< warping kernel; determines warping algorithm
    )
{
    int numGoodPixels = 0;

    typedef typename DestExposureT::MaskedImage DestMaskedImageT;
    typedef typename SrcExposureT::MaskedImage SrcMaskedImageT;
    typedef afwImage::Image<afwMath::Kernel::PixelT> KernelImageT;
    
    // Compute borders; use to prevent applying kernel outside of srcExposure
    int xBorder0 = warpingKernel.getCtrX();
    int yBorder0 = warpingKernel.getCtrY();
    int xBorder1 = warpingKernel.getWidth() - (1 + xBorder0);
    int yBorder1 = warpingKernel.getHeight() - (1 + yBorder0);

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

    // The source image accessor points to (0,0) which corresponds to pixel xBorder0, yBorder0
    // because the accessor points to (0,0) of the kernel rather than the center of the kernel
    const typename DestMaskedImageT::SinglePixel blankPixel(0, 0, edgePixelMask);
    
    std::vector<double> kernelXList(warpingKernel.getWidth());
    std::vector<double> kernelYList(warpingKernel.getHeight());

    // Set each pixel of destExposure's MaskedImage
    lsst::pex::logging::Trace("lsst.afw.math", 4, "Remapping masked image");
    typename DestMaskedImageT::SinglePixel tempPixel(0, 0, 0);
    for (int destY = 0; destY < destHeight; ++destY) {
        afwImage::PointD destPosXY(0.0, afwImage::indexToPosition(destY));
        typename DestMaskedImageT::x_iterator destXIter = destMI.row_begin(destY);
        for (int destX = 0; destX < destWidth; ++destX, ++destXIter) {
            // compute sky position associated with this pixel of remapped MaskedImage
            destPosXY[0] = afwImage::indexToPosition(destX);
            afwImage::PointD raDec = destWcsPtr->xyToRaDec(destPosXY);            
            
            // compute associated pixel position on source MaskedImage
            afwImage::PointD srcPosXY = srcWcsPtr->raDecToXY(raDec);

            // Compute new corresponding position on source image and break it into integer and fractional
            // parts; the latter is used to compute the remapping kernel.
            std::vector<double> fracOrigPix(2);
            int srcX = afwImage::positionToIndex(fracOrigPix[0], srcPosXY[0]);
            int srcY = afwImage::positionToIndex(fracOrigPix[1], srcPosXY[1]);
            
            // If location is too near the edge of the source, or off the source, mark the dest as edge
            if ((srcX - xBorder0 < 0) || (srcX + xBorder1 >= srcWidth) 
                || (srcY - yBorder0 < 0) || (srcY + yBorder1 >= srcHeight)) {
                // skip this pixel
                *destXIter = blankPixel;
//                lsst::pex::logging::Trace("lsst.afw.math", 5, "skipping pixel at destX=%d; destY=%d",
//                    destX, destY);
                continue;
            }
            
            ++numGoodPixels;

            // Compute warped pixel
            double kSum = warpingKernel.computeVectors(kernelXList, kernelYList, false);
            typename SrcMaskedImageT::const_xy_locator srcLoc = srcMI.xy_at(srcX, srcY);
            *destXIter = afwMath::convolveAtAPoint<DestMaskedImageT, SrcMaskedImageT>(srcLoc, kernelXList, kernelYList);

            // Correct intensity due to relative pixel spatial scale and kernel sum
            double multFac = destWcsPtr->pixArea(destPosXY) / (srcWcsPtr->pixArea(srcPosXY) * kSum);
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
        afwImage::Exposure<DESTIMAGEPIXELT, afwImage::MaskPixel, afwImage::VariancePixel> &destExposure, \
        afwImage::Exposure<SRCIMAGEPIXELT, afwImage::MaskPixel, afwImage::VariancePixel> const &srcExposure, \
        SeparableKernel const &warpingKernel);


warpExposureFuncByType(float, boost::uint16_t)
warpExposureFuncByType(double, boost::uint16_t)
warpExposureFuncByType(float, int)
warpExposureFuncByType(double, int)
warpExposureFuncByType(float, float)
warpExposureFuncByType(double, float)
warpExposureFuncByType(double, double)
