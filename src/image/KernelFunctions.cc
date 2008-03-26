// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of members of lsst::fw::kernel
 *
 * \ingroup fw
 */
#include <stdexcept>

#include <boost/format.hpp>

#include <lsst/mwi/exceptions.h>
#include <lsst/fw/KernelFunctions.h>

/**
 * \brief Print the pixel values of a Kernel to std::cout
 *
 * Rows increase upward and columns to the right; thus the lower left pixel is (0,0).
 *
 * \ingroup fw
 */
void
lsst::fw::kernel::printKernel(
    lsst::fw::Kernel const &kernel,     ///< the kernel
    double x,                           ///< x at which to evaluate kernel
    double y,                           ///< y at which to evaluate kernel
    bool doNormalize,                   ///< if true, normalize kernel
    std::string pixelFmt                ///< format for pixel values
) {
    typedef lsst::fw::Kernel::PixelT PixelT;
    typedef lsst::fw::Image<PixelT>::pixel_accessor imageAccessorType;
    PixelT kSum;
    lsst::fw::Image<PixelT> kImage = kernel.computeNewImage(kSum, x, y, doNormalize);
    imageAccessorType imRow = kImage.origin();
    imRow.advance(0, kImage.getRows()-1);
    for (unsigned int row=0; row < kImage.getRows(); ++row, imRow.prev_row()) {
        imageAccessorType imCol = imRow;
        for (unsigned int col = 0; col < kImage.getCols(); ++col, imCol.next_col()) {
            std::cout << boost::format(pixelFmt) % (*imCol) << " ";
        }
        std::cout << std::endl;
    }
    if (doNormalize && std::abs(static_cast<double>(kSum) - 1.0) > 1.0e-5) {
        std::cout << boost::format("Warning! Sum of all pixels = %9.5f != 1.0\n") % kSum;
    }
    std::cout << std::endl;
}

//}}}
