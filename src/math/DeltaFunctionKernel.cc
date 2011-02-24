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
 * @brief Definitions of DeltaFunctionKernel member functions.
 *
 * @ingroup fw
 */
#include <sstream>
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
/**
 * @brief Construct a spatially invariant DeltaFunctionKernel
 *
 * @throw pexExcept::InvalidParameterException if active pixel is off the kernel
 */
afwMath::DeltaFunctionKernel::DeltaFunctionKernel(
    int width,              ///< kernel size (columns)
    int height,             ///< kernel size (rows)
    afwGeom::PointI const &point   ///< index of active pixel (where 0,0 is the lower left corner)
) :
    Kernel(width, height, 0),
    _pixel(point)
{
    if (point.getX() < 0 || point.getX() >= width || point.getY() < 0 || point.getY() >= height) {
        std::ostringstream os;
        os << "point (" << point.getX() << ", " << point.getY() << ") lies outside "
            << width << "x" << height << " sized kernel";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
}

afwMath::Kernel::Ptr afwMath::DeltaFunctionKernel::clone() const {
    afwMath::Kernel::Ptr retPtr(new afwMath::DeltaFunctionKernel(this->getWidth(), this->getHeight(),
        this->_pixel));
    retPtr->setCtrX(this->getCtrX());
    retPtr->setCtrY(this->getCtrY());
    return retPtr;
}

double afwMath::DeltaFunctionKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool,
    double,
    double
) const {
    if (image.getDimensions() != this->getDimensions()) {
        std::ostringstream os;
        os << "image dimensions = ( " << image.getWidth() << ", " << image.getHeight()
            << ") != (" << this->getWidth() << ", " << this->getHeight() << ") = kernel dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }

    const int pixelX = getPixel().getX(); // active pixel in Kernel
    const int pixelY = getPixel().getY();

    image = 0;
    *image.xy_at(pixelX, pixelY) = 1;

    return 1;
}

std::string afwMath::DeltaFunctionKernel::toString(std::string const& prefix) const {
    const int pixelX = getPixel().getX(); // active pixel in Kernel
    const int pixelY = getPixel().getY();

    std::ostringstream os;            
    os << prefix << "DeltaFunctionKernel:" << std::endl;
    os << prefix << "Pixel (c,r) " << pixelX << "," << pixelY << ")" << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}
