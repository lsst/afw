// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definition of convolveWithInterpolation and helper functions declared in ConvolveImage.h
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "boost/cstdint.hpp" 

#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/deprecated.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetail = lsst::afw::math::detail;

/**
 * @brief Convolve an Image or MaskedImage with a spatially varying Kernel using linear interpolation
 * (if it is sufficiently accurate, else fall back to brute force computation).
 *
 * This is a low-level convolution function that does not set edge pixels.
 *
 * The algorithm is as follows:
 * - divide the image into regions whose size is no larger than maxInterpolationDistance
 * - for each region:
 *   - convolve it using convolveRegionWithRecursiveInterpolation (which see)
 *
 * Note that this routine will also work with spatially invariant kernels, but not efficiently.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if outImage is not the same size as inImage
 */
template <typename OutImageT, typename InImageT>
void mathDetail::convolveWithInterpolation(
        OutImageT &outImage,        ///< convolved image = inImage convolved with kernel
        InImageT const &inImage,    ///< input image
        lsst::afw::math::Kernel const &kernel,  ///< convolution kernel
        lsst::afw::math::ConvolutionControl const &convolutionControl)  ///< convolution control parameters
{
    if (outImage.getDimensions() != inImage.getDimensions()) {
        std::ostringstream os;
        os << "outImage dimensions = ( "
            << outImage.getWidth() << ", " << outImage.getHeight()
            << ") != (" << inImage.getWidth() << ", " << inImage.getHeight() << ") = inImage dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }

    // compute full region covering good area of output image
    afwGeom::BoxI bbox(afwGeom::Point2I::make(kernel.getCtrX(), kernel.getCtrY()),
        afwGeom::Extent2I::make(
            outImage.getWidth() + 1 - kernel.getWidth(),
            outImage.getHeight() + 1 - kernel.getHeight()));
    KernelImagesForRegion fullRegion(KernelImagesForRegion(kernel.clone(), bbox,
        convolutionControl.getDoNormalize()));

    // divide full region into subregions small enough to interpolate over
    int nx = bbox.getWidth() / convolutionControl.getMaxInterpolationDistance();
    int ny = bbox.getHeight() / convolutionControl.getMaxInterpolationDistance();
    pexLog::TTrace<4>("lsst.afw.math.convolve",
        "convolveWithInterpolation: divide into %d x %d subregions", nx, ny);

    std::vector<KernelImagesForRegion> subregionList = fullRegion.getSubregions(nx, ny);
    
    for (std::vector<KernelImagesForRegion>::iterator regionPtr = subregionList.begin();
        regionPtr != subregionList.end(); ++regionPtr) {
        convolveRegionWithRecursiveInterpolation(outImage, inImage, *regionPtr,
            convolutionControl.getMaxInterpolationError());
    }            
}

/**
 * @brief Convolve a region of an Image or MaskedImage with a spatially varying Kernel
 * using recursion and interpolation.
 *
 * This is a low-level convolution function that does not set edge pixels.
 *
 * The algorithm is:
 * - if the region is too small:
 *     - solve it with brute force
 * - if interpolation is acceptable (using KernelImagesForRegion::isInterpolationOk):
 *     - convolve with an interpolated kernel
 * - else:
 *     - divide the region into four subregions and call this subroutine for on each subregion
 *
 * Note that this routine will also work with spatially invariant kernels, but not efficiently.
 *
 * @warning This is a low-level routine that performs no bounds checking.
 */
template <typename OutImageT, typename InImageT>
void mathDetail::convolveRegionWithRecursiveInterpolation(
        OutImageT &outImage,        ///< convolved image = inImage convolved with kernel
        InImageT const &inImage,    ///< input image
        KernelImagesForRegion const &region,    ///< kernel image region over which to convolve
        double maxInterpolationError)           ///< maximum allowed error in computing the kernel image
            ///< at any pixel via linear interpolation
{
    if (afwGeom::any(region.getBBox().getDimensions().lt(
        afwGeom::Extent2I::make(region.getMinInterpSize())))) {
        // region too small for interpolation; convolve using brute force
        afwMath::Kernel::ConstPtr kernelPtr = region.getKernel();
        afwGeom::BoxI const bbox = kernelPtr->growBBox(region.getBBox());
        OutImageT outView(OutImageT(outImage, afwGeom::convertToImage(bbox)));
        InImageT inView(InImageT(inImage, afwGeom::convertToImage(bbox)));
        mathDetail::convolveWithBruteForce(outView, inView, *kernelPtr, region.getDoNormalize());
    } else if (region.isInterpolationOk(maxInterpolationError)) {
        // convolve region using linear interpolation
        KernelImagesForRegion::List rgnList = region.getSubregions();
        for (KernelImagesForRegion::List::const_iterator regionPtr = rgnList.begin();
            regionPtr != rgnList.end(); ++regionPtr) {
            convolveRegionWithInterpolation(outImage, inImage, *regionPtr);
        }
    } else {
        // linear interpolation wasn't good enough; divide region into 2x2 subregions and recurse on those
        KernelImagesForRegion::List rgnList = region.getSubregions();
        for (KernelImagesForRegion::List::const_iterator regionPtr = rgnList.begin();
            regionPtr != rgnList.end(); ++regionPtr) {
            convolveRegionWithRecursiveInterpolation(outImage, inImage, *regionPtr, maxInterpolationError);
        }
    }
}

/**
 * @brief Convolve a region of an Image or MaskedImage with a spatially varying Kernel using interpolation.
 *
 * This is a low-level convolution function that does not set edge pixels.
 *
 * @warning: this is a low-level routine that performs no bounds checking.
 */
template <typename OutImageT, typename InImageT>
void mathDetail::convolveRegionWithInterpolation(
        OutImageT &outImage,        ///< convolved image = inImage convolved with kernel
        InImageT const &inImage,    ///< input image
        KernelImagesForRegion const &region)    ///< kernel image region over which to convolve
{
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef typename InImageT::const_xy_locator InLocator;
    typedef KernelImagesForRegion::Image KernelImage;
    typedef KernelImage::const_xy_locator KernelLocator;
    
    afwMath::Kernel::ConstPtr kernelPtr = region.getKernel();
    std::pair<int, int> const kernelDimensions(kernelPtr->getDimensions());
    KernelImage leftKernelImage(*(region.getImage(KernelImagesForRegion::BOTTOM_LEFT)), true);
    KernelImage rightKernelImage(*(region.getImage(KernelImagesForRegion::BOTTOM_RIGHT)), true);
    KernelImage leftDeltaKernelImage(kernelDimensions);
    KernelImage rightDeltaKernelImage(kernelDimensions);
    KernelImage deltaKernelImage(kernelDimensions);  // interpolated in x
    KernelImage kernelImage(leftKernelImage, true);  // final interpolated kernel image

    afwGeom::BoxI const outBBox = region.getBBox();
    afwGeom::BoxI const inBBox = kernelPtr->growBBox(outBBox);
    
    double xfrac = 1.0 / static_cast<double>(outBBox.getWidth());
    double yfrac = 1.0 / static_cast<double>(outBBox.getHeight());
    afwMath::scaledPlus(leftDeltaKernelImage, 
         yfrac,  leftKernelImage,
        -yfrac, *region.getImage(KernelImagesForRegion::TOP_LEFT));
    afwMath::scaledPlus(rightDeltaKernelImage,
        yfrac, rightKernelImage,
        -yfrac, *region.getImage(KernelImagesForRegion::TOP_RIGHT));


    // note: it might be slightly more efficient to compute locators directly on outImage and inImage,
    // without making views; however, using views seems a bit simpler and safer
    // (less likelihood of accidentally straying out of the region)
    OutImageT outView(OutImageT(outImage, afwGeom::convertToImage(outBBox)));
    InImageT inView(InImageT(inImage, afwGeom::convertToImage(inBBox)));
    KernelLocator const kernelLocator = kernelImage.xy_at(0, 0);
    
    // the loop is a bit odd for efficiency: the initial value of kernelImage, leftKernelImage and
    // rightKernelImage are set when they are allocated, so they are not computed in the loop
    // until after the convolution; to save cpu cycles they are not computed at all in the last iteration.
    for (int row = 0; ; ) {
        afwMath::scaledPlus(deltaKernelImage, xfrac, leftKernelImage, -xfrac, rightKernelImage);
        OutXIterator outIter = outImage.row_begin(row);
        OutXIterator const outEnd = outImage.row_end(row);
        InLocator inLocator = inImage.xy_at(row, 0);
        for ( ; outIter != outEnd; ++outIter, ++inLocator.x()) {
            *outIter = afwMath::convolveAtAPoint<OutImageT, InImageT>(inLocator, kernelLocator,
                kernelPtr->getWidth(), kernelPtr->getHeight());
            kernelImage += deltaKernelImage;
        }
        row += 1;
        if (row >= outView.getHeight()) break;
        leftKernelImage += leftDeltaKernelImage;
        rightKernelImage += rightDeltaKernelImage;
        kernelImage <<= leftKernelImage;
    }
}

/*
 * Explicit instantiation
 *
 * Modelled on ConvolveImage.cc
 */
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define NL /* */
// Instantiate Image or MaskedImage versions
#define INSTANTIATE_IM_OR_MI(IMGMACRO, OUTPIXTYPE, INPIXTYPE) \
    template void mathDetail::convolveWithInterpolation( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::Kernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::convolveRegionWithRecursiveInterpolation( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, KernelImagesForRegion const&, double); NL \
    template void mathDetail::convolveRegionWithInterpolation( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, KernelImagesForRegion const&);
// Instantiate both Image and MaskedImage versions
#define INSTANTIATE(OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(IMAGE,       OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(MASKEDIMAGE, OUTPIXTYPE, INPIXTYPE)

INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, boost::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, boost::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(boost::uint16_t, boost::uint16_t)
