// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of DeltaFunctionKernel member functions and explicit instantiations of the class.
 *
 * \ingroup fw
 */
#include <vector>

#include <vw/Image.h>

#include <lsst/pex/exceptions.h>
#include <lsst/afw/math/Kernel.h>

/**
 * \brief Construct a spatially invariant DeltaFunctionKernel
 */
lsst::afw::math::DeltaFunctionKernel::DeltaFunctionKernel(int pixelCol,
                                                   int pixelRow,
                                                   unsigned int cols,
                                                   unsigned int rows)
    : Kernel(cols, rows, 0),
      _pixel(pixelCol, pixelRow)
{
#if 0
    std::vector<double> params;
    params.push_back(pixelCol);
    params.push_back(pixelRow);
    setRHLParameters(params);
#endif

    const int c = pixelCol + getCtrCol(); // column and
    const int r = pixelRow + getCtrRow(); //     row in image

    if (c < 0 || c >= static_cast<int>(getCols()) ||
        r < 0 || r >= static_cast<int>(getRows())) {
        throw lsst::pex::exceptions::InvalidParameter("Pixel lies outside image");
    }
}

void lsst::afw::math::DeltaFunctionKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    PixelT &imSum,
    double x,
    double y,
    bool
) const {
    typedef lsst::afw::image::Image<PixelT>::pixel_accessor pixelAccessor;
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }

    image *= 0;
    pixelAccessor imPtr = image.origin();

    const int c = getPixel().first + getCtrCol(); // column and
    const int r = getPixel().second + getCtrRow(); //     row in image

    imPtr.advance(c, r);
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
