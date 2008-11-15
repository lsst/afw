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

/**
 * @brief Construct a spatially varying SeparableKernel, replicating a spatial function once per kernel function parameter
 */
lsst::afw::math::SeparableKernel::SeparableKernel(
    int width,
    int height,
    KernelFunction const& kernelColFunction,
    KernelFunction const& kernelRowFunction,
    Kernel::SpatialFunction const& spatialFunction)
:
    Kernel(width, height, kernelColFunction.getNParameters() + kernelRowFunction.getNParameters(), spatialFunction),
    _kernelColFunctionPtr(kernelColFunction.copy()),
    _kernelRowFunctionPtr(kernelRowFunction.copy()),
    _localColList(width),
    _localRowList(height)
{}

/**
 * @brief Construct a spatially varying SeparableKernel
 *
 * @throw lsst::pex::exceptions::InvalidParameter if the length of spatialFunctionList != # kernel function parameters.
 */
lsst::afw::math::SeparableKernel::SeparableKernel(
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
    if (kernelColFunction.getNParameters() + kernelRowFunction.getNParameters() != spatialFunctionList.size()) {
        throw lsst::pex::exceptions::InvalidParameter(
            "Length of spatialFunctionList does not match # of kernel function params");
    }
}

double lsst::afw::math::SeparableKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.getDimensions() != this->getDimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    
    double imSum = basicComputeVectors(_localColList, _localRowList, doNormalize);

    for (int y = 0; y != image.getHeight(); ++y) {
        lsst::afw::image::Image<PixelT>::x_iterator imPtr = image.row_begin(y);
        for (std::vector<PixelT>::iterator colIter = _localColList.begin();
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
 * @throw lsst::pex::exceptions::InvalidParameter if colList or rowList is the wrong size
 */
double lsst::afw::math::SeparableKernel::computeVectors(
    std::vector<PixelT> &colList,   ///< column vector
    std::vector<PixelT> &rowList,   ///< row vector
    bool doNormalize,   ///< normalize the image (so sum of each is 1)?
    double x,   ///< x (column position) at which to compute spatial function
    double y    ///< y (row position) at which to compute spatial function
) const {
    if (static_cast<int>(colList.size()) != this->getWidth() || static_cast<int>(rowList.size()) != this->getHeight()) {
        throw lsst::pex::exceptions::InvalidParameter("colList and/or rowList are the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    
    return basicComputeVectors(colList, rowList, doNormalize);
}

/**
 * @brief Get a deep copy of the col kernel function
 */
lsst::afw::math::SeparableKernel::KernelFunctionPtr lsst::afw::math::SeparableKernel::getKernelColFunction(
) const {
    return _kernelColFunctionPtr->copy();
}

/**
 * @brief Get a deep copy of the row kernel function
 */
lsst::afw::math::SeparableKernel::KernelFunctionPtr lsst::afw::math::SeparableKernel::getKernelRowFunction(
) const {
    return _kernelRowFunctionPtr->copy();
}

std::string lsst::afw::math::SeparableKernel::toString(std::string prefix) const {
    std::ostringstream os;
    os << prefix << "SeparableKernel:" << std::endl;
    os << prefix << "..x (width) function: " << (_kernelColFunctionPtr ? _kernelColFunctionPtr->toString() : "None") << std::endl;
    os << prefix << "..y (rows) function: " << (_kernelRowFunctionPtr ? _kernelRowFunctionPtr->toString() : "None") << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
};

std::vector<double> lsst::afw::math::SeparableKernel::getKernelParameters() const {
    std::vector<double> allParams = _kernelColFunctionPtr->getParameters();
    std::vector<double> yParams = _kernelRowFunctionPtr->getParameters();
    std::copy(yParams.begin(), yParams.end(), std::back_inserter(allParams));
    return allParams;
}

//
// Protected Member Functions
//

void lsst::afw::math::SeparableKernel::setKernelParameter(unsigned int ind, double value) const {
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
 * Warning: no range checking!
 */
double lsst::afw::math::SeparableKernel::basicComputeVectors(
    std::vector<PixelT> &colList,   ///< column vector
    std::vector<PixelT> &rowList,   ///< row vector
    bool doNormalize   ///< normalize the arrays (so sum of each is 1)?
) const {
    double colSum = 0.0;
    double colFuncValue;
    std::vector<PixelT>::iterator colIter = colList.begin();
    double xOffset = - static_cast<double>(this->getCtrX());

    for (double x = xOffset; colIter != colList.end(); ++colIter, x += 1.0) {
        colFuncValue = (*_kernelColFunctionPtr)(x);
        *colIter = colFuncValue;
        colSum += colFuncValue;
    }

    double rowSum = 0.0;
    double rowFuncValue;
    std::vector<PixelT>::iterator rowIter = rowList.begin();
    double yOffset = - static_cast<double>(this->getCtrY());
    for (double y = yOffset; rowIter != rowList.end(); ++rowIter, y += 1.0) {
        rowFuncValue = (*_kernelRowFunctionPtr)(y);
        *rowIter = rowFuncValue;
        rowSum += rowFuncValue;
    }

    double imSum = 0;                   // sum of image pixels
    if (doNormalize) {
        colIter = colList.begin();
        for ( ; colIter != colList.end(); ++colIter) {
            *colIter /= colSum;
        }

        rowIter = rowList.begin();
        for ( ; rowIter != rowList.end(); ++rowIter) {
            *rowIter /= rowSum;
        }
        imSum = 1;
    } else {
        imSum = colSum * rowSum;
    }

    return imSum;
}
