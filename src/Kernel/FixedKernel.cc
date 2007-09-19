// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of FixedKernel member functions and explicit instantiations of the class.
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <stdexcept>

#include <lsst/mwi/exceptions/Exception.h>
#include <lsst/fw/Kernel.h>
#include <vw/Image.h>

//
// Constructors
//

/**
 * \brief Construct an empty FixedKernel of size 0x0
 */
template<typename PixelT>
lsst::fw::FixedKernel<PixelT>::FixedKernel()
:
    Kernel<PixelT>(),
    _image(),
    _sum(0) {
}

/**
 * \brief Construct a FixedKernel from an image
 */
template<typename PixelT>
lsst::fw::FixedKernel<PixelT>::FixedKernel(
    Image<PixelT> const &image)     ///< image for kernel
:
    Kernel<PixelT>(image.getCols(), image.getRows(), 0),
    _image(image),
    _sum(vw::sum_of_channel_values(*(image.getIVwPtr()))) {
}

//
// Member Functions
//

template<typename PixelT>
void lsst::fw::FixedKernel<PixelT>::computeImage(
    Image<PixelT> &image,
    PixelT &imSum,
    double x,
    double y,
    bool doNormalize
) const {
    typedef typename Image<PixelT>::pixel_accessor pixelAccessor;
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::mwi::exceptions::InvalidParameter("image is the wrong size");
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

template<typename PixelT>
std::vector<double> lsst::fw::FixedKernel<PixelT>::getCurrentKernelParameters() const {
    return std::vector<double>(0);
}

//
// Protected Member Functions
//

template<typename PixelT>
void lsst::fw::FixedKernel<PixelT>::basicSetKernelParameters(std::vector<double> const &params) const {
    if (params.size() > 0) {
        throw lsst::mwi::exceptions::InvalidParameter("FixedKernel has no kernel parameters");
    }
}

//
// Private Member Functions
//

// Explicit instantiations
template class lsst::fw::FixedKernel<double>;
