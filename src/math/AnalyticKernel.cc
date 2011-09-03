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
 * @brief Definitions of AnalyticKernel member functions.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

/**
 * @brief Construct an empty spatially invariant AnalyticKernel of size 0x0
 */
afwMath::AnalyticKernel::AnalyticKernel()
:
    Kernel(),
    _kernelFunctionPtr()
{}

/**
 * @brief Construct a spatially invariant AnalyticKernel,
 * or a spatially varying AnalyticKernel where the spatial model
 * is described by one function (that is cloned to give one per analytic function parameter).
 */
afwMath::AnalyticKernel::AnalyticKernel(
    int width,  ///< width of kernel
    int height, ///< height of kernel
    KernelFunction const &kernelFunction,   ///< kernel function; a deep copy is made
    Kernel::SpatialFunction const &spatialFunction  ///< spatial function;
        ///< one deep copy is made for each kernel function parameter;
        ///< if omitted or set to Kernel::NullSpatialFunction then the kernel is spatially invariant
) :
    Kernel(width, height, kernelFunction.getNParameters(), spatialFunction),
    _kernelFunctionPtr(kernelFunction.clone())
{}

/**
 * @brief Construct a spatially varying AnalyticKernel, where the spatial model
 * is described by a list of functions (one per analytic function parameter).
 *
 * @throw lsst::pex::exceptions::InvalidParameterException
 *        if the length of spatialFunctionList != # kernel function parameters.
 */
afwMath::AnalyticKernel::AnalyticKernel(
    int width,  ///< width of kernel
    int height, ///< height of kernel
    KernelFunction const &kernelFunction,   ///< kernel function; a deep copy is made
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList  ///< list of spatial functions,
        ///< one per kernel function parameter; a deep copy is made of each function
) :
    Kernel(width, height, spatialFunctionList),
    _kernelFunctionPtr(kernelFunction.clone())
{
    if (kernelFunction.getNParameters() != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelFunction.getNParameters() = " << kernelFunction.getNParameters()
            << " != " << spatialFunctionList.size() << " = " << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
}

afwMath::Kernel::Ptr afwMath::AnalyticKernel::clone() const {
    afwMath::Kernel::Ptr retPtr;
    if (this->isSpatiallyVarying()) {
        retPtr.reset(new afwMath::AnalyticKernel(this->getWidth(), this->getHeight(),
            *(this->_kernelFunctionPtr), this->_spatialFunctionList));
    } else {
        retPtr.reset(new afwMath::AnalyticKernel(this->getWidth(), this->getHeight(),
            *(this->_kernelFunctionPtr)));
    }
    retPtr->setCtrX(this->getCtrX());
    retPtr->setCtrY(this->getCtrY());
    return retPtr;
}

double afwMath::AnalyticKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize,
    double xPos,
    double yPos
) const {
#if 0
    if (image.getDimensions() != this->getDimensions()) {
        std::ostringstream os;
        os << "image dimensions = ( " << image.getWidth() << ", " << image.getHeight()
            << ") != (" << this->getWidth() << ", " << this->getHeight() << ") = kernel dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
#endif
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(xPos, yPos);
    }

    double xOffset = -this->getCtrX();
    double yOffset = -this->getCtrY();

    double imSum = 0;
    for (int y = 0; y != image.getHeight(); ++y) {
        double const fy = y + yOffset;
        afwImage::Image<Pixel>::x_iterator ptr = image.row_begin(y);
        for (int x = 0; x != image.getWidth(); ++x, ++ptr) {
            double const fx = x + xOffset;
            Pixel const pixelVal = (*_kernelFunctionPtr)(fx, fy);
            *ptr = pixelVal;
            imSum += pixelVal;
        }
    }
    if (doNormalize) {
        if (imSum == 0) {
            throw LSST_EXCEPT(pexExcept::OverflowErrorException, "Cannot normalize; kernel sum is 0");
        }
        image /= imSum;
        imSum = 1;
    }

    return imSum;
}

/**
 * @brief Get a deep copy of the kernel function
 */
afwMath::AnalyticKernel::KernelFunctionPtr afwMath::AnalyticKernel::getKernelFunction(
) const {
    return _kernelFunctionPtr->clone();
}

std::string afwMath::AnalyticKernel::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << prefix << "AnalyticKernel:" << std::endl;
    os << prefix << "..function: " << (_kernelFunctionPtr ? _kernelFunctionPtr->toString() : "None")
        << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}

std::vector<double> afwMath::AnalyticKernel::getKernelParameters() const {
    return _kernelFunctionPtr->getParameters();
}

//
// Protected Member Functions
//
void afwMath::AnalyticKernel::setKernelParameter(unsigned int ind, double value) const {
    _kernelFunctionPtr->setParameter(ind, value);
}
