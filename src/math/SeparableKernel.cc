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

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

/**
 * @brief Construct a spatially varying SeparableKernel,
 *  copying one spatial function once per kernel function parameter
 */
afwMath::SeparableKernel::SeparableKernel(
    int width,
    int height,
    KernelFunction const& kernelColFunction,
    KernelFunction const& kernelRowFunction,
    Kernel::SpatialFunction const& spatialFunction)
:
    Kernel(width, height, kernelColFunction.getNParameters() + kernelRowFunction.getNParameters(),
        spatialFunction),
    _kernelColFunctionPtr(kernelColFunction.copy()),
    _kernelRowFunctionPtr(kernelRowFunction.copy()),
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
    int width,
    int height,
    KernelFunction const& kernelColFunction,
    KernelFunction const& kernelRowFunction,
    std::vector<Kernel::SpatialFunctionPtr> const& spatialFunctionList)
:
    Kernel(width, height, spatialFunctionList),
    _kernelColFunctionPtr(kernelColFunction.copy()),
    _kernelRowFunctionPtr(kernelRowFunction.copy()),
    _localColList(width),
    _localRowList(height)
{
    if (kernelColFunction.getNParameters() + kernelRowFunction.getNParameters()
        != spatialFunctionList.size()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
            "Length of spatialFunctionList does not match # of kernel function params");
    }
}

double afwMath::SeparableKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.getDimensions() != this->getDimensions()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
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
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "colList and/or rowList are the wrong size");
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
    return _kernelColFunctionPtr->copy();
}

/**
 * @brief Get a deep copy of the row kernel function
 */
afwMath::SeparableKernel::KernelFunctionPtr afwMath::SeparableKernel::getKernelRowFunction(
) const {
    return _kernelRowFunctionPtr->copy();
}

std::string afwMath::SeparableKernel::toString(std::string prefix) const {
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
 * Warning: no range checking!
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
