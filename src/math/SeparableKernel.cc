// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of SeparableKernel member functions.
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <algorithm>
#include <iterator>

#include "vw/Image.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

/**
 * \brief Construct an empty spatially invariant SeparableKernel of size 0x0
 */
lsst::afw::math::SeparableKernel::SeparableKernel()
:
    Kernel(),
    _kernelColFunctionPtr(),
    _kernelRowFunctionPtr()
{}

/**
 * \brief Construct a spatially invariant SeparableKernel
 */
lsst::afw::math::SeparableKernel::SeparableKernel(
    KernelFunction const &kernelColFunction,
    KernelFunction const &kernelRowFunction,
    unsigned int cols,
    unsigned int rows)
:
    Kernel(cols, rows, kernelColFunction.getNParameters() + kernelRowFunction.getNParameters()),
    _kernelColFunctionPtr(kernelColFunction.copy()),
    _kernelRowFunctionPtr(kernelRowFunction.copy()),
    _localColList(cols),
    _localRowList(rows)
{}

/**
 * \brief Construct a spatially varying SeparableKernel, replicating a spatial function once per kernel function parameter
 */
lsst::afw::math::SeparableKernel::SeparableKernel(
    KernelFunction const &kernelColFunction,
    KernelFunction const &kernelRowFunction,
    unsigned int cols,
    unsigned int rows,
    Kernel::SpatialFunction const &spatialFunction)
:
    Kernel(cols, rows, kernelColFunction.getNParameters() + kernelRowFunction.getNParameters(), spatialFunction),
    _kernelColFunctionPtr(kernelColFunction.copy()),
    _kernelRowFunctionPtr(kernelRowFunction.copy()),
    _localColList(cols),
    _localRowList(rows)
{}

/**
 * \brief Construct a spatially varying SeparableKernel
 *
 * \throw lsst::pex::exceptions::InvalidParameter if the length of spatialFunctionList != # kernel function parameters.
 */
lsst::afw::math::SeparableKernel::SeparableKernel(
    KernelFunction const &kernelColFunction,
    KernelFunction const &kernelRowFunction,
    unsigned int cols,
    unsigned int rows,
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)
:
    Kernel(cols, rows, spatialFunctionList),
    _kernelColFunctionPtr(kernelColFunction.copy()),
    _kernelRowFunctionPtr(kernelRowFunction.copy()),
    _localColList(cols),
    _localRowList(rows)
{
    if (kernelColFunction.getNParameters() + kernelRowFunction.getNParameters() != spatialFunctionList.size()) {
        throw lsst::pex::exceptions::InvalidParameter(
            "Length of spatialFunctionList does not match # of kernel function params");
    }
}

void lsst::afw::math::SeparableKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    PixelT &imSum,
    bool doNormalize,
    double x,
    double y
) const {
    typedef lsst::afw::image::Image<PixelT>::pixel_accessor pixelAccessor;
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    
    basicComputeVectors(_localColList, _localRowList, imSum, doNormalize);

    std::vector<PixelT>::iterator rowIter = _localRowList.begin();
    pixelAccessor imRow = image.origin();
    for ( ; rowIter != _localRowList.end(); ++rowIter, imRow.next_row()) {
        pixelAccessor imCol = imRow;
        std::vector<PixelT>::iterator colIter = _localColList.begin();
        for ( ; colIter != _localColList.end(); ++colIter, imCol.next_col()) {
            *imCol = (*colIter) * (*rowIter);
        }
    }
}

/**
 * @brief Compute the column and row arrays in place, where kernel(col, row) = colList(col) * rowList(row)
 *
 * x, y are ignored if there is no spatial function.
 *
 * @throw lsst::pex::exceptions::InvalidParameter if colList or rowList is the wrong size
 */
void lsst::afw::math::SeparableKernel::computeVectors(
    std::vector<PixelT> &colList,   ///< column vector
    std::vector<PixelT> &rowList,   ///< row vector
    PixelT &imSum,  ///< sum of image pixels (output)
    bool doNormalize,   ///< normalize the image (so sum of each is 1)?
    double x,   ///< x (column position) at which to compute spatial function
    double y    ///< y (row position) at which to compute spatial function
) const {
    if ((colList.size() != this->getCols()) || (rowList.size() != this->getRows())) {
        throw lsst::pex::exceptions::InvalidParameter("colList and/or rowList are the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    
    return basicComputeVectors(colList, rowList, imSum, doNormalize);
}

/**
 * \brief Get a deep copy of the col kernel function
 */
lsst::afw::math::SeparableKernel::KernelFunctionPtr lsst::afw::math::SeparableKernel::getKernelColFunction(
) const {
    return _kernelColFunctionPtr->copy();
}

/**
 * \brief Get a deep copy of the row kernel function
 */
lsst::afw::math::SeparableKernel::KernelFunctionPtr lsst::afw::math::SeparableKernel::getKernelRowFunction(
) const {
    return _kernelRowFunctionPtr->copy();
}

std::string lsst::afw::math::SeparableKernel::toString(std::string prefix) const {
    std::ostringstream os;
    os << prefix << "SeparableKernel:" << std::endl;
    os << prefix << "..x (cols) function: " << (_kernelColFunctionPtr ? _kernelColFunctionPtr->toString() : "None") << std::endl;
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
    const unsigned int nColParams = _kernelColFunctionPtr->getNParameters();
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
void lsst::afw::math::SeparableKernel::basicComputeVectors(
    std::vector<PixelT> &colList,   ///< column vector
    std::vector<PixelT> &rowList,   ///< row vector
    PixelT &imSum,  ///< sum of image pixels (output)
    bool doNormalize   ///< normalize the arrays (so sum of each is 1)?
) const {
    double colSum = 0.0;
    double colFuncValue;
    std::vector<PixelT>::iterator colIter = colList.begin();
    double xOffset = - static_cast<double>(this->getCtrCol());
    for (double x = xOffset; colIter != colList.end(); ++colIter, x += 1.0) {
        colFuncValue = (*_kernelColFunctionPtr)(x);
        *colIter = colFuncValue;
        colSum += colFuncValue;
    }

    double rowSum = 0.0;
    double rowFuncValue;
    std::vector<PixelT>::iterator rowIter = rowList.begin();
    double yOffset = - static_cast<double>(this->getCtrRow());
    for (double y = yOffset; rowIter != rowList.end(); ++rowIter, y += 1.0) {
        rowFuncValue = (*_kernelRowFunctionPtr)(y);
        *rowIter = rowFuncValue;
        rowSum += rowFuncValue;
    }

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
}
