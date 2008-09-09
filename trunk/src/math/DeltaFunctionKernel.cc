// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definitions of DeltaFunctionKernel member functions.
 *
 * @ingroup fw
 */
#include <vector>

#include "vw/Image.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

/**
 * @brief Construct a spatially invariant DeltaFunctionKernel
 */
lsst::afw::math::DeltaFunctionKernel::DeltaFunctionKernel(
    unsigned int pixelCol,  ///< active pixel colum (0 is left column)
    unsigned int pixelRow,  ///< active pixel row (0 is bottom row)
    unsigned int cols,  ///< kernel size (columns)
    unsigned int rows)  ///< kernel size (rows)
:
    Kernel(cols, rows, 0),
    _pixel(pixelCol, pixelRow)
{
#if 0
    std::vector<double> params;
    params.push_back(pixelCol);
    params.push_back(pixelRow);
    setRHLParameters(params);
#endif

    if ((pixelCol >= cols) || (pixelRow >= rows)) {
        throw lsst::pex::exceptions::InvalidParameter("Active pixel lies outside image");
    }
}

void lsst::afw::math::DeltaFunctionKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    PixelT &imSum,
    bool doNormalize,
    double x,
    double y
) const {
    typedef lsst::afw::image::Image<PixelT>::pixel_accessor pixelAccessor;
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }

    image *= 0;
    pixelAccessor imPtr = image.origin();

    const int pixelCol = getPixel().first; // active pixel in Kernel
    const int pixelRow = getPixel().second;

    imPtr.advance(pixelCol, pixelRow);
    *imPtr = imSum = 1;
}

std::string lsst::afw::math::DeltaFunctionKernel::toString(std::string prefix) const {
    const int pixelCol = getPixel().first; // active pixel in Kernel
    const int pixelRow = getPixel().second;

    std::ostringstream os;            
    os << prefix << "DeltaFunctionKernel:" << std::endl;
    os << prefix << "Pixel (c,r) " << pixelCol << "," << pixelRow << ")" << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
};
