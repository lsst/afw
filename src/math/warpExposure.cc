// -*- LSST-C++ -*- // fixed format comment for emacs
/**
 * @file
 *
 * @ingroup afw
 *
 * @brief Implementation of the templated utility function, warpExposure, for
 * Astrometric Image Remapping for LSST.  Declared in warpExposure.h.
 *
 * @author Nicole M. Silvestri, University of Washington
 */

#include <string>
#include <vector>
#include <cmath>

#include <boost/cstdint.hpp> 
#include <boost/format.hpp> 

#include "lsst/daf/base/DataProperty.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h" 
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

/**
 * @brief Remap an Exposure to a new WCS.
 *
 * kernelType is one of:
 * * nearest: nearest neighbor (not yet implemented)
 *   Good noise conservation, bad aliasing issues. Best used for weight maps.
 *   Kernel size must be 3x3
 * * bilinear:
 *   Good for undersampled data fast (not yet implemented)
 *   Kernel size must be 3x3
 * * lanczos: 2-d Lanczos function:
 *   Accurate but slow.
 *   (x, y) = sinc(pi x') sinc(pi x' / n) sinc(pi y') sinc(pi y' / n)
 *   with n = (min(kernelHeight, kernelWidth) - 1)/2
 *
 * For pixels in remapExposure that cannot be computed because their data comes from pixels that are too close
 * to (or off of) the edge of origExposure.
 * * The image and variance are set to 0
 * * The mask bit EDGE is set, if present, else the mask pixel is set to 0
 * * the total number of all such pixels is returned
 *
 * @return the number valid pixels in remapExposure (the rest are off the edge).
 *
 * @throw lsst::pex::exceptions::InvalidParameter error if kernelType is not one of the
 * types listed above.
 * @throw lsst::pex::exceptions::InvalidParameter error if kernelWidth != 3 or kernelHeight != 3
 * and kernelType is nearest or bilinear.
 *
 * Algorithm:
 *
 * For each integer pixel position in the remapped Exposure:
 * * The associated sky coordinates are determined using the remapped WCS.
 * * The associated pixel position on origExposure is determined using the original WCS.
 * * A remapping kernel is computed based on the fractional part of the pixel position on origExposure
 * * The remapping kernel is applied to origExposure at the integer portion of the pixel position
 *   to compute the remapped pixel value
 * * The flux-conserving factor is determined from the original and new WCS.
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
template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT> 
int afwMath::warpExposure(
    afwImage::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> &remapExposure,      ///< remapped exposure
    afwImage::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> const &origExposure, ///< original exposure
    std::string const &kernelType,  ///< kernel type (see main function docs for more info)
    const int kernelWidth,   ///< kernel size - columns
    const int kernelHeight    ///< kernel size - height
    )
{
    int numGoodPixels = 0;

    typedef afwImage::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> MaskedImageT;
    typedef afwImage::Image<afwMath::Kernel::PixelT> KernelImageT;
    
    // Create remapping AnalyticKernel of desired type and size.
    typedef typename afwMath::AnalyticKernel::KernelFunctionPtr FunctionPtr;
    FunctionPtr akPtrFcn; ///< analytic kernel pointer function

    if (kernelType == "nearest") { 
        // Nearest Neighbor Interpolation. Not yet implemented.
        if (kernelWidth != 3 || kernelHeight != 3) {
            throw lsst::pex::exceptions::InvalidParameter(
                boost::format("Kernel size must be 3x3 for kernelType %s") % kernelType);
        }
    } else if (kernelType == "bilinear") { 
        // Bi-Linear Interpolation. Not yet implemented.
        if (kernelWidth != 3 || kernelHeight != 3) {
            throw lsst::pex::exceptions::InvalidParameter(
                boost::format("Kernel size must be 3x3 for kernelType %s") % kernelType);
        }
    } else if (kernelType == "lanczos") { 
        // 2D Lanczos resampling kernel
        int order = (std::min(kernelHeight, kernelWidth) - 1)/2;
        akPtrFcn = FunctionPtr(new afwMath::LanczosFunction2<afwMath::Kernel::PixelT>(order));
    } else {
        throw lsst::pex::exceptions::InvalidParameter(
            boost::format("Invalid kernelType %s") % kernelType);
    }
    if (!akPtrFcn) {
        throw lsst::pex::exceptions::InvalidParameter(
            boost::format("kernelType %s not yet implemented") % kernelType);
    }

    afwMath::AnalyticKernel remapKernel(kernelWidth, kernelHeight, *akPtrFcn);
    lsst::pex::logging::Trace("lsst.afw.math", 3,
        boost::format("Created analytic kernel of type=%s; width=%d; height=%d")
        % kernelType % kernelWidth % kernelHeight);
    
    // Compute kernel extent; use to prevent applying kernel outside of origExposure
    int xBorder0 = remapKernel.getCtrX();
    int yBorder0 = remapKernel.getCtrY();
    int xBorder1 = remapKernel.getWidth() - 1 - xBorder0;
    int yBorder1 = remapKernel.getHeight() - 1 - yBorder0;

    // Create a blank kernel image of the appropriate size and get a pixel locator to it.
    KernelImageT kImage(kernelWidth, kernelHeight);
    typename KernelImageT::const_xy_locator kLoc = kImage.xy_at(0, 0);

    // Get the original MaskedImage and a pixel accessor to it.
    MaskedImageT origMI = origExposure.getMaskedImage();
    const int origWidth = origMI.getWidth();
    const int origHeight = origMI.getHeight();
    typename afwImage::Wcs::Ptr origWcsPtr = origExposure.getWcs();
    lsst::pex::logging::Trace("lsst.afw.math", 3,
        boost::format("orig image width=%d; height=%d") % origWidth % origHeight);

    // Get the remapped MaskedImage and the remapped wcs.
    MaskedImageT remapMI = remapExposure.getMaskedImage();
    typename afwImage::Wcs::Ptr remapWcsPtr = remapExposure.getWcs();
   
    // Conform mask plane names of remapped MaskedImage to match original
    remapMI.getMask()->conformMaskPlanes(origMI.getMask()->getMaskPlaneDict());
    
    // Make a pixel mask from the EDGE bit, if available (0 if not available)
    const MaskPixelT edgePixelMask = origMI.getMask()->getPlaneBitMask("EDGE");
    lsst::pex::logging::Trace("lsst.afw.math", 3, boost::format("edgePixelMask=0x%X") % edgePixelMask);
    
    const int remapWidth = remapMI.getWidth();
    const int remapHeight = remapMI.getHeight();
    lsst::pex::logging::Trace("lsst.afw.math", 3,
        boost::format("remap image width=%d; height=%d") % remapWidth % remapHeight);

    // The original image accessor points to (0,0) which corresponds to pixel xBorder0, yBorder0
    // because the accessor points to (0,0) of the kernel rather than the center of the kernel
    const typename MaskedImageT::SinglePixel blankPixel(0, 0, edgePixelMask);

    // Set each pixel of remapExposure's MaskedImage
    lsst::pex::logging::Trace("lsst.afw.math", 4, "Remapping masked image");
    for (int remapY = 0; remapY < remapHeight; ++remapY) {
        afwImage::PointD remapPosXY(0.0, afwImage::indexToPosition(remapY));
        typename MaskedImageT::x_iterator remapPtr = remapMI.row_begin(remapY);
        for (int remapX = 0; remapX < remapWidth; ++remapX, ++remapPtr) {
            // compute sky position associated with this pixel of remapped MaskedImage
            remapPosXY[0] = afwImage::indexToPosition(remapX);
            afwImage::PointD raDec = remapWcsPtr->xyToRaDec(remapPosXY);            
            
            // compute associated pixel position on original MaskedImage
            afwImage::PointD origPosXY = origWcsPtr->raDecToXY(raDec);

            // Compute new corresponding position on original image and break it into integer and fractional
            // parts; the latter is used to compute the remapping kernel.
            std::vector<double> fracOrigPix(2);
            int origX = afwImage::positionToIndex(fracOrigPix[0], origPosXY[0]);
            int origY = afwImage::positionToIndex(fracOrigPix[1], origPosXY[1]);
            
            // Check new position before applying it
            if ((origX - xBorder0 < 0) || (origX + xBorder1 >= origWidth) 
                || (origY - yBorder0 < 0) || (origY + yBorder1 >= origHeight)) {
                // skip this pixel
                *remapPtr = blankPixel;
//                lsst::pex::logging::Trace("lsst.afw.math", 5, "skipping pixel at remapX=%d; remapY=%d",
//                    remapX, remapY);
                continue;
            }
            
            ++numGoodPixels;

            // New original pixel position is usable, advance to it
            typename MaskedImageT::const_xy_locator origMILoc = origMI.xy_at(origX, origY);
            afwImage::PointD origXY(origX, origY);   

            // Compute new kernel image based on fractional pixel position
            remapKernel.setKernelParameters(fracOrigPix); 
            double kSum = 0;
            remapKernel.computeImage(kImage, kSum, false);
            
            // Determine the intensity multipler due to relative pixel scale and kernel sum
            double multFac = remapWcsPtr->pixArea(remapPosXY) / (origWcsPtr->pixArea(origXY) * kSum);
           
            // Apply remapping kernel to original MaskedImage to compute remapped pixel
            *remapPtr = afwMath::apply<MaskedImageT, MaskedImageT>(origMILoc, kLoc, kernelWidth, kernelHeight);

            // Correct output by relative area of input and output pixels
            remapPtr.image() *= static_cast<ImagePixelT>(multFac);
            remapPtr.variance() *= static_cast<ImagePixelT>(multFac * multFac);

        } // for remapX loop
    } // for remapY loop      
    return numGoodPixels;
} // warpExposure


// /** 
//  * @brief Remap an Exposure to a new WCS.  
//  *
//  * This version takes a remapped Exposure's WCS (probably a copy of an existing WCS), the requested size
//  * of the remapped MaskedImage, the exposure to be remapped (original Exposure), and the remapping kernel
//  * information (kernel type and size).
//  *
//  * @return the final remapped Exposure
//  */
// template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT> 
// afwImage::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> afwMath::warpExposure(
//     int &numGoodPixels, ///< number of pixels that were not computed because they were too close to the edge
//                         ///< (or off the edge) of origExposure
//     afwImage::Wcs const &remapWcs,  ///< remapped exposure's WCS
//     const int remapWidth,            ///< remapped exposure size - columns
//     const int remapHeight,            ///< remapped exposure size - height
//     afwImage::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> const &origExposure, ///< original exposure 
//     std::string const kernelType,   ///< kernel type (see main function docs for more info)
//     const int kernelWidth,   ///< kernel size - columns
//     const int kernelHeight    ///< kernel size - height
//     )
// {
//     afwImage::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> remapMaskedImage(remapWidth, remapHeight);
//     afwImage::Exposure<ImagePixelT, MaskPixelT, VariancePixelT> remapExposure(remapMaskedImage, remapWcs);
// 
//     numGoodPixels = afwMath::warpExposure(
//         remapExposure, origExposure, kernelType, kernelWidth, kernelHeight); 
// 
//     return remapExposure;
// 
// } // warpExposure

/************************************************************************************************************/
//
// Explicit instantiations
//
typedef float imagePixelType;

template
int afwMath::warpExposure(
    afwImage::Exposure<boost::uint16_t, afwImage::MaskPixel, afwImage::VariancePixel> &remapExposure,
    afwImage::Exposure<boost::uint16_t, afwImage::MaskPixel, afwImage::VariancePixel> const &origExposure,
    std::string const &kernelType, int kernelWidth, int kernelHeight);

template
int afwMath::warpExposure(
    afwImage::Exposure<float, afwImage::MaskPixel, afwImage::VariancePixel> &remapExposure,
    afwImage::Exposure<float, afwImage::MaskPixel, afwImage::VariancePixel> const &origExposure,
    std::string const &kernelType, int kernelWidth, int kernelHeight);

template
int afwMath::warpExposure(
    afwImage::Exposure<double, afwImage::MaskPixel, afwImage::VariancePixel> &remapExposure,
    afwImage::Exposure<double, afwImage::MaskPixel, afwImage::VariancePixel> const &origExposure,
    std::string const &kernelType, int kernelWidth, int kernelHeight);

// template
// afwImage::Exposure<boost::uint16_t, afwImage::MaskPixel, afwImage::VariancePixel> afwMath::warpExposure(
//     int &numGoodPixels,
//     afwImage::Wcs const &remapWcs,
//     const int remapWidth,       
//     const int remapHeight,       
//     afwImage::Exposure<boost::uint16_t, afwImage::MaskPixel, afwImage::VariancePixel> const &origExposure,
//     std::string const kernelType, 
//     const int kernelWidth,  
//     const int kernelHeight);
// 
// template
// afwImage::Exposure<imagePixelType, afwImage::MaskPixel, afwImage::VariancePixel> afwMath::warpExposure(
//     int &numGoodPixels,
//     afwImage::Wcs const &remapWcs,
//     const int remapWidth,       
//     const int remapHeight,       
//     afwImage::Exposure<imagePixelType, afwImage::MaskPixel, afwImage::VariancePixel> const &origExposure,
//     std::string const kernelType, 
//     const int kernelWidth,  
//     const int kernelHeight);
// template
// afwImage::Exposure<double, afwImage::MaskPixel, afwImage::VariancePixel> afwMath::warpExposure(
//     int &numGoodPixels,
//     afwImage::Wcs const &remapWcs,
//     const int remapWidth,       
//     const int remapHeight,       
//     afwImage::Exposure<double, afwImage::MaskPixel, afwImage::VariancePixel> const &origExposure,
//     std::string const kernelType, 
//     const int kernelWidth,  
//     const int kernelHeight);
