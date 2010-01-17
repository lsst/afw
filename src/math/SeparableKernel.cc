// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definitions of SeparableKernel member functions.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <algorithm>
#include <iterator>
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

/**
 * @brief Construct an empty spatially invariant SeparableKernel of size 0x0
 */
afwMath::SeparableKernel::SeparableKernel()
:
    Kernel(),
    _kernelColFunctionPtr(),
    _kernelRowFunctionPtr(),
    _localColList(0),
    _localRowList(0)
{}

/**
 * @brief Construct a spatially invariant SeparableKernel, or a spatially varying SeparableKernel
 * that uses the same functional form to model each function parameter.
 */
afwMath::SeparableKernel::SeparableKernel(
    int width,  ///< width of kernel
    int height, ///< height of kernel
    KernelFunction const& kernelColFunction,    ///< kernel column function
    KernelFunction const& kernelRowFunction,    ///< kernel row function
    Kernel::SpatialFunction const& spatialFunction  ///< spatial function;
        ///< one deep copy is made for each kernel column and row function parameter;
        ///< if omitted or set to Kernel::NullSpatialFunction then the kernel is spatially invariant
) :
    Kernel(width, height, kernelColFunction.getNParameters() + kernelRowFunction.getNParameters(),
        spatialFunction),
    _kernelColFunctionPtr(kernelColFunction.clone()),
    _kernelRowFunctionPtr(kernelRowFunction.clone()),
    _localColList(width),
    _localRowList(height)
{}

/**
 * @brief Construct a spatially varying SeparableKernel
 *
 * @throw lsst::pex::exceptions::InvalidParameterException
 *  if the length of spatialFunctionList != # kernel function parameters.
 */
afwMath::SeparableKernel::SeparableKernel(
    int width,  ///< width of kernel
    int height, ///< height of kernel
    KernelFunction const& kernelColFunction,    ///< kernel column function
    KernelFunction const& kernelRowFunction,    ///< kernel row function
    std::vector<Kernel::SpatialFunctionPtr> const& spatialFunctionList  ///< list of spatial functions,
        ///< one per kernel column and row function parameter; a deep copy is made of each function
) :
    Kernel(width, height, spatialFunctionList),
    _kernelColFunctionPtr(kernelColFunction.clone()),
    _kernelRowFunctionPtr(kernelRowFunction.clone()),
    _localColList(width),
    _localRowList(height)
{
    if (kernelColFunction.getNParameters() + kernelRowFunction.getNParameters()
        != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelColFunction.getNParameters() + kernelRowFunction.getNParameters() = "
            << kernelColFunction.getNParameters() << " + " << kernelRowFunction.getNParameters()
            << " != " << spatialFunctionList.size() << " = " << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
}

afwMath::Kernel::Ptr afwMath::SeparableKernel::clone() const {
    afwMath::Kernel::Ptr retPtr;
    if (this->isSpatiallyVarying()) {
        retPtr.reset(new afwMath::SeparableKernel(this->getWidth(), this->getHeight(),
            *(this->_kernelColFunctionPtr), *(this->_kernelRowFunctionPtr), this->_spatialFunctionList));
    } else {
        retPtr.reset(new afwMath::SeparableKernel(this->getWidth(), this->getHeight(),
            *(this->_kernelColFunctionPtr), *(this->_kernelRowFunctionPtr)));
    }
    retPtr->setCtrX(this->getCtrX());
    retPtr->setCtrY(this->getCtrY());
    return retPtr;
}

double afwMath::SeparableKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize,
    double xPos,
    double yPos
) const {
    if (image.getDimensions() != this->getDimensions()) {
        std::ostringstream os;
        os << "image dimensions = ( " << image.getWidth() << ", " << image.getHeight()
            << ") != (" << this->getWidth() << ", " << this->getHeight() << ") = kernel dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(xPos, yPos);
    }
    
    double imSum = basicComputeVectors(_localColList, _localRowList, doNormalize);

    for (int y = 0; y != image.getHeight(); ++y) {
        afwImage::Image<Pixel>::x_iterator imPtr = image.row_begin(y);
        for (std::vector<Pixel>::iterator colIter = _localColList.begin();
             colIter != _localColList.end(); ++colIter, ++imPtr) {
            *imPtr = (*colIter)*_localRowList[y];
        }
    }
    
    return imSum;
}

/**
 * @brief Compute the column and row arrays in place, where kernel(col, row) = colList(col) * rowList(row)
 *
 * x, y are ignored if there is no spatial function.
 *
 * @return the kernel sum (1.0 if doNormalize true)
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if colList or rowList is the wrong size
 * @throw lsst::pex::exceptions::OverflowErrorException if doNormalize is true and the kernel sum is
 * exactly 0
 */
double afwMath::SeparableKernel::computeVectors(
    std::vector<Pixel> &colList,   ///< column vector
    std::vector<Pixel> &rowList,   ///< row vector
    bool doNormalize,   ///< normalize the image (so sum of each is 1)?
    double x,   ///< x (column position) at which to compute spatial function
    double y    ///< y (row position) at which to compute spatial function
) const {
    if (static_cast<int>(colList.size()) != this->getWidth()
        || static_cast<int>(rowList.size()) != this->getHeight()) {
        std::ostringstream os;
        os << "colList.size(), rowList.size() = ("
            << colList.size() << ", " << rowList.size()
            << ") != ("<< this->getWidth() << ", " << this->getHeight()
            << ") = " << "kernel dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    
    return basicComputeVectors(colList, rowList, doNormalize);
}

/**
 * @brief Get a deep copy of the col kernel function
 */
afwMath::SeparableKernel::KernelFunctionPtr afwMath::SeparableKernel::getKernelColFunction(
) const {
    return _kernelColFunctionPtr->clone();
}

/**
 * @brief Get a deep copy of the row kernel function
 */
afwMath::SeparableKernel::KernelFunctionPtr afwMath::SeparableKernel::getKernelRowFunction(
) const {
    return _kernelRowFunctionPtr->clone();
}

std::string afwMath::SeparableKernel::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << prefix << "SeparableKernel:" << std::endl;
    os << prefix << "..x (width) function: "
        << (_kernelColFunctionPtr ? _kernelColFunctionPtr->toString() : "None") << std::endl;
    os << prefix << "..y (rows) function: "
        << (_kernelRowFunctionPtr ? _kernelRowFunctionPtr->toString() : "None") << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
};

std::vector<double> afwMath::SeparableKernel::getKernelParameters() const {
    std::vector<double> allParams = _kernelColFunctionPtr->getParameters();
    std::vector<double> yParams = _kernelRowFunctionPtr->getParameters();
    std::copy(yParams.begin(), yParams.end(), std::back_inserter(allParams));
    return allParams;
}

//
// Protected Member Functions
//

void afwMath::SeparableKernel::setKernelParameter(unsigned int ind, double value) const {
    unsigned int const nColParams = _kernelColFunctionPtr->getNParameters();
    if (ind < nColParams) {
        _kernelColFunctionPtr->setParameter(ind, value);
    } else {
        _kernelRowFunctionPtr->setParameter(ind - nColParams, value);
    }
}

//
// Private Member Functions
//

/**
 * @brief Compute the column and row arrays in place, where kernel(col, row) = colList(col) * rowList(row)
 *
 * @return the kernel sum (1.0 if doNormalize true)
 *
 * Warning: the length of colList and rowList are not verified!
 *
 * @throw lsst::pex::exceptions::OverflowErrorException if doNormalize is true and the kernel sum is
 * exactly 0
 */
double afwMath::SeparableKernel::basicComputeVectors(
    std::vector<Pixel> &colList,   ///< column vector
    std::vector<Pixel> &rowList,   ///< row vector
    bool doNormalize   ///< normalize the arrays (so sum of each is 1)?
) const {
    double colSum = 0.0;
    double xArg = - static_cast<double>(this->getCtrX());
    for (std::vector<Pixel>::iterator colIter = colList.begin();
        colIter != colList.end(); ++colIter, ++xArg) {
        double colFuncValue = (*_kernelColFunctionPtr)(xArg);
        *colIter = colFuncValue;
        colSum += colFuncValue;
    }

    double rowSum = 0.0;
    double yArg = - static_cast<double>(this->getCtrY());
    for (std::vector<Pixel>::iterator rowIter = rowList.begin();
        rowIter != rowList.end(); ++rowIter, ++yArg) {
        double rowFuncValue = (*_kernelRowFunctionPtr)(yArg);
        *rowIter = rowFuncValue;
        rowSum += rowFuncValue;
    }

    double imSum = colSum * rowSum;
    if (doNormalize) {
        if ((colSum == 0) || (rowSum == 0)) {
            throw LSST_EXCEPT(pexExcept::OverflowErrorException, "Cannot normalize; kernel sum is 0");
        }
        for (std::vector<Pixel>::iterator colIter = colList.begin(); colIter != colList.end(); ++colIter) {
            *colIter /= colSum;
        }

        for (std::vector<Pixel>::iterator rowIter = rowList.begin(); rowIter != rowList.end(); ++rowIter) {
            *rowIter /= rowSum;
        }
        imSum = 1.0;
    }
    return imSum;
}
