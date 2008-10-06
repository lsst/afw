// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definitions of DeltaFunctionKernel member functions.
 *
 * @ingroup fw
 */
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

/**
 * @brief Construct a spatially invariant DeltaFunctionKernel
 */
lsst::afw::math::DeltaFunctionKernel::DeltaFunctionKernel(
    int pixelX,                       ///< active pixel colum (0 is left column)
    int pixelY,                       ///< active pixel row (0 is bottom row)
    int width,                          ///< kernel size (columns)
    int height)                         ///< kernel size (rows)
:
    Kernel(width, height, 0),
    _pixel(pixelX, pixelY)
{
#if 0
    std::vector<double> params;
    params.push_back(pixelX);
    params.push_back(pixelY);
    setRHLParameters(params);
#endif

    if ((pixelX >= width) || (pixelY >= height)) {
        throw lsst::pex::exceptions::InvalidParameter("Active pixel lies outside image");
    }
}

double lsst::afw::math::DeltaFunctionKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.dimensions() != this->dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }

    const int pixelX = getPixel().first; // active pixel in Kernel
    const int pixelY = getPixel().second;

    image = 0;
    *image.xy_at(pixelX, pixelY) = 1;

    return 1;
}

std::string lsst::afw::math::DeltaFunctionKernel::toString(std::string prefix) const {
    const int pixelX = getPixel().first; // active pixel in Kernel
    const int pixelY = getPixel().second;

    std::ostringstream os;            
    os << prefix << "DeltaFunctionKernel:" << std::endl;
    os << prefix << "Pixel (c,r) " << pixelX << "," << pixelY << ")" << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
};
