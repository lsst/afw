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

namespace ex = lsst::pex::exceptions;

/**
 * @brief Construct a spatially invariant DeltaFunctionKernel
 */
lsst::afw::math::DeltaFunctionKernel::DeltaFunctionKernel(
    int width,                          ///< kernel size (columns)
    int height,                         ///< kernel size (rows)
    lsst::afw::image::PointI point      //< Active pixel
                                                         )
:
    Kernel(width, height, 0),
    _pixel(point.getX(), point.getY())
{
    if (point.getX() < 0 || point.getX() >= width || point.getY() < 0 || point.getY() >= height) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Active pixel lies outside image");
    }
}

double lsst::afw::math::DeltaFunctionKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.getDimensions() != this->getDimensions()) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "image is the wrong size");
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
