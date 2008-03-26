// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of DeltaFunctionKernel member functions and explicit instantiations of the class.
 *
 * \ingroup fw
 */
#include <vw/Image.h>

#include <lsst/mwi/exceptions.h>
#include <lsst/fw/Kernel.h>

/**
 * \brief Construct a spatially invariant DeltaFunctionKernel
 */
lsst::fw::DeltaFunctionKernel::DeltaFunctionKernel(int pixelCol,
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
        throw lsst::mwi::exceptions::InvalidParameter("Pixel lies outside image");
    }
}

void lsst::fw::DeltaFunctionKernel::computeImage(
    Image<PixelT> &image,
    PixelT &imSum,
    double x,
    double y,
    bool
) const {
    typedef Image<PixelT>::pixel_accessor pixelAccessor;
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::mwi::exceptions::InvalidParameter("image is the wrong size");
    }

    image *= 0;
    pixelAccessor imPtr = image.origin();

    const int c = getPixel().first + getCtrCol(); // column and
    const int r = getPixel().second + getCtrRow(); //     row in image

    imPtr.advance(c, r);
    *imPtr = imSum = 1;
}
