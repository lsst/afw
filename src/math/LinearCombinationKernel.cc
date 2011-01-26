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
 * @brief Definitions of LinearCombinationKernel member functions.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/LocalKernel.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

/**
 * @brief Construct an empty LinearCombinationKernel of size 0x0
 */
afwMath::LinearCombinationKernel::LinearCombinationKernel()
:
    Kernel(),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(),
    _isDeltaFunctionBasis(false)
{ }

/**
 * @brief Construct a spatially invariant LinearCombinationKernel
 */
afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to const) basis kernels
    std::vector<double> const &kernelParameters) ///< kernel coefficients
:
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), kernelList.size()),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(kernelParameters),
    _isDeltaFunctionBasis(false)
{
    if (kernelList.size() != kernelParameters.size()) {
        std::ostringstream os;
        os << "kernelList.size() = " << kernelList.size()
            << " != " << kernelParameters.size() << " = " << "kernelParameters.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

/**
 * @brief Construct a spatially varying LinearCombinationKernel, where the spatial model
 * is described by one function (that is cloned to give one per basis kernel).
 */
afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to const) basis kernels
    Kernel::SpatialFunction const &spatialFunction)  ///< spatial function;
        ///< one deep copy is made for each basis kernel
:
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), kernelList.size(), spatialFunction),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(std::vector<double>(kernelList.size())),
    _isDeltaFunctionBasis(false)
{
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

/**
 * @brief Construct a spatially varying LinearCombinationKernel, where the spatial model
 * is described by a list of functions (one per basis kernel).
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if the length of spatialFunctionList != # kernels
 */
afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to const) kernels
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)
        ///< list of spatial functions, one per basis kernel
:
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), spatialFunctionList),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(std::vector<double>(kernelList.size())),
    _isDeltaFunctionBasis(false)
{
    if (kernelList.size() != spatialFunctionList.size()) {
        std::ostringstream os;
        os << "kernelList.size() = " << kernelList.size()
            << " != " << spatialFunctionList.size() << " = " << "spatialFunctionList.size()";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    checkKernelList(kernelList);
    _setKernelList(kernelList);
}

afwMath::Kernel::Ptr afwMath::LinearCombinationKernel::clone() const {
    Kernel::Ptr retPtr;
    if (this->isSpatiallyVarying()) {
        retPtr.reset(new afwMath::LinearCombinationKernel(this->_kernelList, this->_spatialFunctionList));
    } else {
        retPtr.reset(new afwMath::LinearCombinationKernel(this->_kernelList, this->_kernelParams));
    }
    retPtr->setCtrX(this->getCtrX());
    retPtr->setCtrY(this->getCtrY());
    return retPtr;
}

/**
 * @brief Check that all kernels have the same size and center and that none are spatially varying
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if the check fails
 */
void afwMath::LinearCombinationKernel::checkKernelList(const KernelList &kernelList) const {
    if (kernelList.size() < 1) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "kernelList has no elements");
    }

    int ctrX = kernelList[0]->getCtrX();
    int ctrY = kernelList[0]->getCtrY();

    for (unsigned int ii = 0; ii < kernelList.size(); ++ii) {
        if (kernelList[ii]->getDimensions() != kernelList[0]->getDimensions()) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                (boost::format("kernel %d has different size than kernel 0") % ii).str());
        }
        if ((ctrX != kernelList[ii]->getCtrX()) || (ctrY != kernelList[ii]->getCtrY())) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                (boost::format("kernel %d has different center than kernel 0") % ii).str());
        }
        if (kernelList[ii]->isSpatiallyVarying()) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                (boost::format("kernel %d is spatially varying") % ii).str());
        }
    }
}

double afwMath::LinearCombinationKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.getDimensions() != this->getDimensions()) {
        std::ostringstream os;
        os << "image dimensions = ( " << image.getWidth() << ", " << image.getHeight()
            << ") != (" << this->getWidth() << ", " << this->getHeight() << ") = kernel dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if (this->isSpatiallyVarying()) {
        this->computeKernelParametersFromSpatialModel(this->_kernelParams, x, y);
    }

    image = 0.0;
    double imSum = 0.0;
    std::vector<afwImage::Image<Pixel>::Ptr>::const_iterator kImPtrIter = _kernelImagePtrList.begin();
    std::vector<double>::const_iterator kSumIter = _kernelSumList.begin();
    std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    for ( ; kImPtrIter != _kernelImagePtrList.end(); ++kImPtrIter, ++kSumIter, ++kParIter) {
        image.scaledPlus(*kParIter, **kImPtrIter);
        imSum += (*kSumIter) * (*kParIter);
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
 * @brief Get the fixed basis kernels
 */
afwMath::KernelList const & afwMath::LinearCombinationKernel::getKernelList() const {
    return _kernelList;
}

/**
* @brief Get the sum of the pixels of each fixed basis kernel
*/
std::vector<double> afwMath::LinearCombinationKernel::getKernelSumList() const {
    return _kernelSumList;
}

std::vector<double> afwMath::LinearCombinationKernel::getKernelParameters() const {
    return _kernelParams;
}

/**
* @brief Refactor the kernel as a linear combination of N bases where N is the number of parameters
* for the spatial model.
*
* Refactoring is only possible if all of the following are true:
*  * Kernel is spatially varying
*  * The spatial functions are a linear combination of coefficients (return isLinearCombination() true).
*  * The spatial functions all are the same class (and so have the same functional form)
* Refactoring produces a kernel that is faster to compute only if the number of basis kernels
* is greater than the number of parameters in the spatial model.
*
* Details:
* A spatially varying LinearCombinationKernel consisting of M basis kernels
* and using a spatial model that is a linear combination of N coefficients can be expressed as:
* K(x,y) =   K0 (C00 F0(x,y) + C10 F1(x,y) + C20 F2(x,y) + ... + CN0 FN(x,y))
*          + K1 (C01 F0(x,y) + C11 F1(x,y) + C21 F2(x,y) + ... + CN1 FN(x,y))
*          + K2 (C02 F0(x,y) + C12 F1(x,y) + C22 F2(x,y) + ... + CN2 FN(x,y))
*          + ...
*          + KM (C0M F0(x,y) + C1M F1(x,y) + C2M F2(x,y) + ... + CNM FN(x,y))
*
* This is equivalent to the following linear combination of N basis kernels:
*
*         =      K0' F0(x,y) + K1' F1(x,y) + K2' F2(x,y) + ... + KN' FN(x,y)
*
*           where Ki' = sum over j of Kj Cij
*
* This is what refactor returns provided the required conditions are met. However, the spatial functions
* for the refactored kernel are the same as those for the original kernel (for generality and simplicity)
* with all coefficients equal to 0 except one that is set to 1; hence they are not computed optimally.
*
* Thanks to Kresimir Cosic for inventing or reinventing this useful technique.
*
* @return a shared pointer to new kernel, or empty pointer if refactoring not possible
*/
afwMath::Kernel::Ptr afwMath::LinearCombinationKernel::refactor() const {
    if (!this->isSpatiallyVarying()) {
        return Kernel::Ptr();
    }
    Kernel::SpatialFunctionPtr const firstSpFuncPtr = this->_spatialFunctionList[0];
    if (!firstSpFuncPtr->isLinearCombination()) {
        return Kernel::Ptr();
    }
    
    typedef lsst::afw::image::Image<Kernel::Pixel> KernelImage;
    typedef boost::shared_ptr<KernelImage> KernelImagePtr;
    typedef std::vector<KernelImagePtr> KernelImageList;
    
    // create kernel images for new refactored basis kernels
    int const nSpatialParameters = this->getNSpatialParameters();
    KernelImageList newKernelImagePtrList;
    newKernelImagePtrList.reserve(nSpatialParameters);
    for (int i = 0; i < nSpatialParameters; ++i) {
        KernelImagePtr kernelImagePtr(new KernelImage(this->getWidth(), this->getHeight()));
        newKernelImagePtrList.push_back(kernelImagePtr);
    }
    KernelImage kernelImage(this->getWidth(), this->getHeight());
    std::vector<Kernel::SpatialFunctionPtr>::const_iterator spFuncPtrIter = 
        this->_spatialFunctionList.begin();
    afwMath::KernelList::const_iterator kIter = _kernelList.begin();
    afwMath::KernelList::const_iterator const kEnd = _kernelList.end();
    for ( ; kIter != kEnd; ++kIter, ++spFuncPtrIter) {
        if (typeid(**spFuncPtrIter) != typeid(*firstSpFuncPtr)) {
            return Kernel::Ptr();
        }
    
        (**kIter).computeImage(kernelImage, false);
        for (int i = 0; i < nSpatialParameters; ++i) {
            double spParam = (*spFuncPtrIter)->getParameter(i);
            newKernelImagePtrList[i]->scaledPlus(spParam, kernelImage);
        }
    }
    
    // create new kernel; the basis kernels are fixed kernels computed above
    // and the corresponding spatial model is the same function as the original kernel,
    // but with all coefficients zero except coeff_i = 1.0
    afwMath::KernelList newKernelList;
    newKernelList.reserve(nSpatialParameters);
    KernelImageList::iterator newKImPtrIter = newKernelImagePtrList.begin();
    KernelImageList::iterator const newKImPtrEnd = newKernelImagePtrList.end();
    for ( ; newKImPtrIter != newKImPtrEnd; ++newKImPtrIter) {
        newKernelList.push_back(Kernel::Ptr(new afwMath::FixedKernel(**newKImPtrIter)));
    }
    std::vector<SpatialFunctionPtr> newSpFunctionPtrList;
    for (int i = 0; i < nSpatialParameters; ++i) {
        std::vector<double> newSpParameters(nSpatialParameters, 0.0);
        newSpParameters[i] = 1.0;
        SpatialFunctionPtr newSpFunctionPtr = firstSpFuncPtr->clone();
        newSpFunctionPtr->setParameters(newSpParameters);
        newSpFunctionPtrList.push_back(newSpFunctionPtr);
    }
    LinearCombinationKernel::Ptr refactoredKernel(
        new LinearCombinationKernel(newKernelList, newSpFunctionPtrList));
    refactoredKernel->setCtr(this->getCtr());
    return refactoredKernel;
}

std::string afwMath::LinearCombinationKernel::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << prefix << "LinearCombinationKernel:" << std::endl;
    os << prefix << "..Kernels:" << std::endl;
    for (KernelList::const_iterator i = _kernelList.begin(); i != _kernelList.end(); ++i) {
        os << (*i)->toString(prefix + "\t");
    }
    os << "..parameters: [ ";
    for (std::vector<double>::const_iterator i = _kernelParams.begin(); i != _kernelParams.end(); ++i) {
        if (i != _kernelParams.begin()) os << ", ";
        os << *i;
    }
    os << " ]" << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
}

//
// Protected Member Functions
//
void afwMath::LinearCombinationKernel::setKernelParameter(unsigned int ind, double value) const {
    this->_kernelParams[ind] = value;
}

//
// Private Member Functions
//
/**
 * @brief Set _kernelList by cloning each input kernel and update the kernel image cache.
 */
void afwMath::LinearCombinationKernel::_setKernelList(KernelList const &kernelList) {
    _kernelSumList.clear();
    _kernelImagePtrList.clear();
    _kernelList.clear();
    _isDeltaFunctionBasis = true;
    for (KernelList::const_iterator kIter = kernelList.begin(), kEnd = kernelList.end();
        kIter != kEnd; ++kIter) {
        Kernel::Ptr basisKernelPtr = (*kIter)->clone();
        if (dynamic_cast<afwMath::DeltaFunctionKernel const *>(&(*basisKernelPtr)) == 0) {
            _isDeltaFunctionBasis = false;
        }
        _kernelList.push_back(basisKernelPtr);
        afwImage::Image<Pixel>::Ptr kernelImagePtr(new afwImage::Image<Pixel>(this->getDimensions()));
        _kernelSumList.push_back(basisKernelPtr->computeImage(*kernelImagePtr, false));
        _kernelImagePtrList.push_back(kernelImagePtr);
    }
}

/**
 *  Return a LocalKernel that matches the type requested, at the given location.
 *
 *  The default implementation would support the creation of IMAGE and FOURIER
 *  visitors without derivatives. LinearCombinationKernel (and possibly
 *  AnalyticKernel) can override to provide versions with derivatives.
 */
afwMath::ImageLocalKernel::Ptr afwMath::LinearCombinationKernel::computeImageLocalKernel(
    lsst::afw::geom::Point2D const & location
) const{
    ImageLocalKernel::Image::Ptr imagePtr( 
        new ImageLocalKernel::Image(getWidth(), getHeight())
    );
    lsst::afw::geom::Point2I center = lsst::afw::geom::Point2I(
        getCtrX(), getCtrY()
    );
    computeImage(*imagePtr, true, location.getX(), location.getY());
    std::vector<double> kernelParameters(getNKernelParameters());
    computeKernelParametersFromSpatialModel(
        kernelParameters, 
        location.getX(), location.getY()
    );

    return boost::make_shared<ImageLocalKernel>(
        center,
        kernelParameters,
        imagePtr,
        _kernelImagePtrList
    );
}

