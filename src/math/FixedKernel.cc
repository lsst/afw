// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
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
namespace afwGeom = lsst::afw::geom;
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

    typedef afwImage::Image<Pixel>::x_iterator XIter;
    double imSum = 0.0;
    for (int y = 0; y != image.getHeight(); ++y) {
        for (XIter imPtr = image.row_begin(y), imEnd = image.row_end(y); imPtr != imEnd; ++imPtr) {
            imSum += *imPtr;
        }
    }
    this->_sum = imSum;
}


/**
 * @brief Construct a FixedKernel from a generic Kernel
 */
afwMath::FixedKernel::FixedKernel(
    afwMath::Kernel const& kernel,      ///< Kernel to convert to Fixed
    afwGeom::Point2D const& pos         ///< desired position 
                                 )
:
    Kernel(kernel.getWidth(), kernel.getHeight(), 0),
    _image(kernel.getDimensions()),
    _sum(0) {
    _sum = kernel.computeImage(_image, false, pos[0], pos[1]);
}

//
// Member Functions
//
PTR(afwMath::Kernel) afwMath::FixedKernel::clone() const {
    PTR(afwMath::Kernel) retPtr(new afwMath::FixedKernel(_image));
    retPtr->setCtrX(this->getCtrX());
    retPtr->setCtrY(this->getCtrY());
    return retPtr;
}

double afwMath::FixedKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize,
    double,
    double
) const {
    if (image.getDimensions() != this->getDimensions()) {
        std::ostringstream os;
        os << "image dimensions = ( " << image.getWidth() << ", " << image.getHeight()
            << ") != (" << this->getWidth() << ", " << this->getHeight() << ") = kernel dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }

    double multFactor = 1.0;
    double imSum = this->_sum;
    if (doNormalize) {
        if (imSum == 0) {
            throw LSST_EXCEPT(pexExcept::OverflowErrorException, "Cannot normalize; kernel sum is 0");
        }
        multFactor = 1.0/static_cast<double>(this->_sum);
        imSum = 1.0;
    }

    typedef afwImage::Image<Pixel>::x_iterator XIter;

    for (int y = 0; y != this->getHeight(); ++y) {
        for (XIter imPtr = image.row_begin(y), imEnd = image.row_end(y), kPtr = this->_image.row_begin(y);
            imPtr != imEnd; ++imPtr, ++kPtr) {
            imPtr[0] = multFactor*kPtr[0];
        }
    }

    return imSum;
}

std::string afwMath::FixedKernel::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << prefix << "FixedKernel:" << std::endl;
    os << prefix << "..sum: " << _sum << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}
