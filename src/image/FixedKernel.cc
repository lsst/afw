// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of FixedKernel member functions and explicit instantiations of the class.
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <stdexcept>

#include <vw/Image.h>

#include <lsst/pex/exceptions.h>
#include <lsst/afw/math/Kernel.h>

//
// Constructors
//

/**
 * \brief Construct an empty FixedKernel of size 0x0
 */
lsst::afw::math::FixedKernel::FixedKernel()
:
    Kernel(),
    _image(),
    _sum(0) {
}

/**
 * \brief Construct a FixedKernel from an image
 */
lsst::afw::math::FixedKernel::FixedKernel(
    Image<PixelT> const &image)     ///< image for kernel
:
    Kernel(image.getCols(), image.getRows(), 0),
    _image(image),
    _sum(vw::sum_of_channel_values(*(image.getIVwPtr()))) {
}

//
// Member Functions
//
void lsst::afw::math::FixedKernel::computeImage(
    Image<PixelT> &image,
    PixelT &imSum,
    double x,
    double y,
    bool doNormalize
) const {
    typedef Image<PixelT>::pixel_accessor pixelAccessor;
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }
    double multFactor;
    if (doNormalize) {
        multFactor = 1.0 / static_cast<double>(this->_sum);
        imSum = 1;
    } else {
        multFactor = 1.0;
        imSum = this->_sum;
    }
    pixelAccessor kRow = this->_image.origin();
    pixelAccessor imRow = image.origin();
    for (unsigned int row = 0; row < this->getRows(); ++row) {
        pixelAccessor kCol = kRow;
        pixelAccessor imCol = imRow;
        for (unsigned int col = 0; col < this->getCols(); ++col) {
            *imCol = static_cast<PixelT>(static_cast<double>(*kCol) * multFactor);
            kCol.next_col();
            imCol.next_col();
        }
        kRow.next_row();
        imRow.next_row();
    }
}

std::vector<double> lsst::afw::math::FixedKernel::getCurrentKernelParameters() const {
    return std::vector<double>(0);
}

//
// Protected Member Functions
//

void lsst::afw::math::FixedKernel::basicSetKernelParameters(std::vector<double> const &params) const {
    if (params.size() > 0) {
        throw lsst::pex::exceptions::InvalidParameter("FixedKernel has no kernel parameters");
    }
}

//
// Private Member Functions
//
