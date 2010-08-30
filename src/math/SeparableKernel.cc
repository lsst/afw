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
    _localColList(0), _localRowList(0),
    _kernelX(0), _kernelY(0)
{
    _setKernelXY();
}

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
    _localColList(width), _localRowList(height),
    _kernelX(width), _kernelY(height)    
{
    _setKernelXY();
}

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
    _localColList(width), _localRowList(height),
    _kernelX(width), _kernelY(height)    
{
    if (kernelColFunction.getNParameters() + kernelRowFunction.getNParameters()
        != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelColFunction.getNParameters() + kernelRowFunction.getNParameters() = "
            << kernelColFunction.getNParameters() << " + " << kernelRowFunction.getNParameters()
            << " != " << spatialFunctionList.size() << " = " << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }

    _setKernelXY();
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
}

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
    std::vector<Pixel> &colList,        ///< column vector
    std::vector<Pixel> &rowList,        ///< row vector
    bool doNormalize                    ///< normalize the arrays (so sum of each is 1)?
) const {
    double colSum = 0.0;
    if (_kernelColCache.empty()) {
        for (unsigned int i = 0; i != colList.size(); ++i) {
            double colFuncValue = (*_kernelColFunctionPtr)(_kernelX[i]);
            colList[i] = colFuncValue;
            colSum += colFuncValue;
        }
    } else {
        int const cacheSize = _kernelColCache.size();
        
        int const indx = this->getKernelParameter(0)*cacheSize;

        std::vector<double> &cachedValues = _kernelColCache.at(indx);
        for (unsigned int i = 0; i != colList.size(); ++i) {
            double colFuncValue = cachedValues[i];
            colList[i] = colFuncValue;
            colSum += colFuncValue;
        }
    }

    double rowSum = 0.0;
    if (_kernelRowCache.empty()) {
        for (unsigned int i = 0; i != rowList.size(); ++i) {
            double rowFuncValue = (*_kernelRowFunctionPtr)(_kernelX[i]);
            rowList[i] = rowFuncValue;
            rowSum += rowFuncValue;
        }
    } else {
        int const cacheSize = _kernelRowCache.size();
        
        int const indx = this->getKernelParameter(1)*cacheSize;
        
        std::vector<double> &cachedValues = _kernelRowCache.at(indx);
        for (unsigned int i = 0; i != rowList.size(); ++i) {
            double rowFuncValue = cachedValues[i];
            rowList[i] = rowFuncValue;
            rowSum += rowFuncValue;

#if 0
            if (indx == cacheSize/2) {
                if (::fabs(rowFuncValue - (*_kernelRowFunctionPtr)(_kernelX[i])) > 1e-2) {
                    std::cout << indx << " " << i << " "
                              << rowFuncValue << " "
                              << (*_kernelRowFunctionPtr)(_kernelX[i])
                              << std::endl;
                }
            }
#endif
        }
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

/************************************************************************************************************/
/**
 * Compute a cache of pre-computed Kernels
 */
namespace {
void _computeCache(int const cacheSize,
                   std::vector<double> const& x,
                   afwMath::SeparableKernel::KernelFunctionPtr & func,
                   std::vector<std::vector<double> > *kernelCache)
{
    if (cacheSize <= 0) {
        kernelCache->erase(kernelCache->begin(), kernelCache->end());
        return;
    }

    if (kernelCache[0].size() != x.size()) { // invalid
        kernelCache->erase(kernelCache->begin(), kernelCache->end());
    }

    int const old_cacheSize = kernelCache->size();

    if (cacheSize == old_cacheSize) {
        return;                     // nothing to do
    }

    if (cacheSize < old_cacheSize) {
        kernelCache->erase(kernelCache->begin() + cacheSize, kernelCache->end());
    } else {
        kernelCache->resize(cacheSize);
        for (int i = old_cacheSize; i != cacheSize; ++i) {
            (*kernelCache)[i].resize(x.size());
        }
    }
    //
    // Actually fill the cache
    //
    for (int i = 0; i != cacheSize; ++i) {
        func->setParameter(0, (i + 0.5)/static_cast<double>(cacheSize));
        for (unsigned int j = 0; j != x.size(); ++j) {
            (*kernelCache)[i][j] = (*func)(x[j]);
        }
    }
}
}

void afwMath::SeparableKernel::computeCache(int const cacheSize) {
    afwMath::SeparableKernel::KernelFunctionPtr func;

    func = getKernelColFunction();
    _computeCache(cacheSize, _kernelY, func, &_kernelColCache);

    func = getKernelRowFunction();
    _computeCache(cacheSize, _kernelX, func, &_kernelRowCache);
}

std::vector<std::vector<double> > afwMath::SeparableKernel::_kernelRowCache;
std::vector<std::vector<double> > afwMath::SeparableKernel::_kernelColCache;
