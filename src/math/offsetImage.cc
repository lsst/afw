/**
 * @file
 *
 * Offset an Image (or Mask or MaskedImage) by a constant vector (dx, dy)
 */
#include "lsst/afw/math/offsetImage.h"
#include "lsst/afw/image/ImageUtils.h"

namespace afwImage = lsst::afw::image;

namespace lsst {
namespace afw {
namespace math {

/**
 * @brief Return an image offset by (dx, dy) using the specified algorithm
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if the algorithm's invalid
 */
template<typename ImageT>
typename ImageT::Ptr offsetImage(std::string const& algorithmName, ///< Type of resampling Kernel to use
                                 ImageT const& inImage,            ///< The %image to offset
                                 float dx,                         ///< move the %image this far in the column direction
                                 float dy                          ///< move the %image this far in the row direction
                                ) {
    SeparableKernel::Ptr offsetKernel = makeWarpingKernel(algorithmName);

    typename ImageT::Ptr outImage(new ImageT(inImage, true)); // output image, a deep copy

    std::pair<int, double> deltaX = afwImage::positionToIndex(dx, true); // true => return the std::pair
    std::pair<int, double> deltaY = afwImage::positionToIndex(dy, true);
    //
    // We won't do the integral part of the shift, but we will set [XY]0 correctly
    //
    outImage->setXY0(afwImage::PointI(inImage.getX0() + deltaX.first, inImage.getY0() + deltaY.first));
    //
    // And now the fractional part
    //
    std::vector<double> kernelXList(offsetKernel->getWidth());
    std::vector<double> kernelYList(offsetKernel->getHeight());
    // We seem to have to pass -dx, -dy to setKernelParameters, for reasons RHL doesn't understand
    offsetKernel->setKernelParameters(std::make_pair(-deltaX.second, -deltaY.second));
    // Setting doNormalize to true doesn't work as convolve() recalculates the kernel; #
    (void)offsetKernel->computeVectors(kernelXList, kernelYList, true);

    convolve(*outImage, inImage, *offsetKernel, true);

    return outImage;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
#define INSTANTIATE(TYPE) \
    template afwImage::Image<TYPE>::Ptr offsetImage(std::string const&, afwImage::Image<TYPE> const&, float, float);\
    template afwImage::Image<TYPE>::Ptr offsetImage(std::string const&, afwImage::MaskedImage<TYPE> const&, float, float);

INSTANTIATE(double)
INSTANTIATE(float)

}}}
