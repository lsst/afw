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

namespace pexExcept = lsst::pex::exceptions;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

//
// Constructors
//

/**
 * @brief Construct an empty FixedKernel of size 0x0
 */
afwMath::FixedKernel::FixedKernel()
:
    Kernel(),
    _image(),
    _sum(0) {
}

/**
 * @brief Construct a FixedKernel from an image
 */
afwMath::FixedKernel::FixedKernel(
    afwImage::Image<Pixel> const &image)     ///< image for kernel
:
    Kernel(image.getWidth(), image.getHeight(), 0),
    _image(image, true),
    _sum(0) {

    typedef afwImage::Image<Pixel>::x_iterator x_iterator;
    double imSum = 0.0;
    for (int y = 0; y != image.getHeight(); ++y) {
        for (x_iterator imPtr = image.row_begin(y), imEnd = image.row_end(y); imPtr != imEnd; ++imPtr) {
            imSum += *imPtr;
        }
    }
    this->_sum = imSum;
}

//
// Member Functions
//
afwMath::Kernel::Ptr afwMath::FixedKernel::clone() const {
    afwMath::Kernel::Ptr retPtr(new afwMath::FixedKernel(_image));
    retPtr->setCtrX(this->getCtrX());
    retPtr->setCtrY(this->getCtrY());
    return retPtr;
}

double afwMath::FixedKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.getDimensions() != this->getDimensions()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "image is the wrong size");
    }

    double multFactor = 1.0;
    double imSum = 1.0;
    if (doNormalize) {
        multFactor = 1.0/static_cast<double>(this->_sum);
    } else {
        imSum = this->_sum;
    }

    typedef afwImage::Image<Pixel>::x_iterator x_iterator;

    for (int y = 0; y != this->getHeight(); ++y) {
        x_iterator kPtr = this->_image.row_begin(y);
        for (x_iterator imPtr = image.row_begin(y), imEnd = image.row_end(y);
            imPtr != imEnd; ++imPtr, ++kPtr) {
            imPtr[0] = multFactor*kPtr[0];
        }
    }

    return imSum;
}

std::string afwMath::FixedKernel::toString(std::string prefix) const {
    std::ostringstream os;
    os << prefix << "FixedKernel:" << std::endl;
    os << prefix << "..sum: " << _sum << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
};
