// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definitions of FixedKernel member functions.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <stdexcept>
#include <numeric>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

//
// Constructors
//

/**
 * @brief Construct an empty FixedKernel of size 0x0
 */
lsst::afw::math::FixedKernel::FixedKernel()
:
    Kernel(),
    _image(),
    _sum(0) {
}

/**
 * @brief Construct a FixedKernel from an image
 */
lsst::afw::math::FixedKernel::FixedKernel(
    lsst::afw::image::Image<PixelT> const &image)     ///< image for kernel
:
    Kernel(image.getWidth(), image.getHeight(), 0),
    _image(image, true),
    _sum(0) {
    _sum = std::accumulate(_image.begin(), _image.end(), _sum); // (a loop over y + row_begin())'s a bit faster, but who cares?
}

//
// Member Functions
//
double lsst::afw::math::FixedKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.getDimensions() != this->getDimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }

    double multFactor = 1.0;
    double imSum = 1.0;
    if (doNormalize) {
        multFactor = 1.0/static_cast<double>(this->_sum);
    } else {
        imSum = this->_sum;
    }

    typedef lsst::afw::image::Image<PixelT>::x_iterator x_iterator;

    for (int y = 0; y != this->getHeight(); ++y) {
        x_iterator kRow = this->_image.row_begin(y);
        for (x_iterator imRow = image.row_begin(y), imEnd = image.row_end(y); imRow != imEnd; ++imRow, ++kRow) {
            imRow[0] = multFactor*kRow[0];
        }
    }

    return imSum;
}

std::string lsst::afw::math::FixedKernel::toString(std::string prefix) const {
    std::ostringstream os;
    os << prefix << "FixedKernel:" << std::endl;
    os << prefix << "..sum: " << _sum << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
};
