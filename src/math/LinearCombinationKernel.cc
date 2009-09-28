// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definitions of LinearCombinationKernel member functions.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <stdexcept>
#include <numeric>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace ex = lsst::pex::exceptions;

/**
 * @brief Construct an empty LinearCombinationKernel of size 0x0
 */
lsst::afw::math::LinearCombinationKernel::LinearCombinationKernel()
:
    Kernel(),
    _kernelList(),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams()
{ }

/**
 * @brief Construct a spatially invariant LinearCombinationKernel
 */
lsst::afw::math::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to) kernels
    std::vector<double> const &kernelParameters) ///< kernel coefficients
:
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), kernelList.size()),
    _kernelList(kernelList),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(kernelParameters)
{
    checkKernelList(kernelList);
    _computeKernelImageList();
}

/**
 * @brief Construct a spatially varying LinearCombinationKernel with spatial parameters initialized to 0
 */
lsst::afw::math::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to) kernels
    Kernel::SpatialFunction const &spatialFunction)  ///< spatial function
:
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), kernelList.size(), spatialFunction),
    _kernelList(kernelList),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(std::vector<double>(kernelList.size()))
{
    checkKernelList(kernelList);
    _computeKernelImageList();
}

/**
 * @brief Construct a spatially varying LinearCombinationKernel with the spatially varying parameters specified
 *
 * @throw lsst::pex::exceptions::InvalidParameterException  if the length of spatialFunctionList != # kernels
 */
lsst::afw::math::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to) kernels
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)    ///< list of spatial functions, one per kernel
:
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), spatialFunctionList),
    _kernelList(kernelList),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(std::vector<double>(kernelList.size()))
{
    if (kernelList.size() != spatialFunctionList.size()) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
            "Length of spatialFunctionList does not match length of kernelList");
    }
    checkKernelList(kernelList);
    _computeKernelImageList();
}

double lsst::afw::math::LinearCombinationKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.getDimensions() != this->getDimensions()) {
        throw LSST_EXCEPT(ex::InvalidParameterException,"image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->computeKernelParametersFromSpatialModel(this->_kernelParams, x, y);
    }

    image = 0.0;
    double imSum = 0.0;
    std::vector<lsst::afw::image::Image<PixelT>::Ptr>::const_iterator kImPtrIter = _kernelImagePtrList.begin();
    std::vector<double>::const_iterator kSumIter = _kernelSumList.begin();
    std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    lsst::afw::image::Image<PixelT>::Ptr tmpImage(new lsst::afw::image::Image<PixelT>(image.getDimensions()));
    for ( ; kImPtrIter != _kernelImagePtrList.end(); ++kImPtrIter, ++kSumIter, ++kParIter) {
        image.scaledPlus(*kParIter, **kImPtrIter);
        imSum += (*kSumIter) * (*kParIter);
    }

    if (doNormalize) {
        image /= imSum;
        imSum = 1;
    }

    return imSum;
}

/**
 * @brief Get the fixed basis kernels
 */
lsst::afw::math::LinearCombinationKernel::KernelList const &
lsst::afw::math::LinearCombinationKernel::getKernelList() const {
    return _kernelList;
}
    
/**
 * @brief Check that all kernels have the same size and center and that none are spatially varying
 *
 * @throw lsst::pex::exceptions::InvalidParameterException  if the check fails
 */
void lsst::afw::math::LinearCombinationKernel::checkKernelList(const KernelList &kernelList) const {
    if (kernelList.size() < 1) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "kernelList has no elements");
    }
    
    int ctrX = kernelList[0]->getCtrX();
    int ctrY = kernelList[0]->getCtrY();
    
    for (unsigned int ii = 0; ii < kernelList.size(); ++ii) {
        if (kernelList[ii]->getDimensions() != kernelList[0]->getDimensions()) {
            throw LSST_EXCEPT(ex::InvalidParameterException,
                (boost::format("kernel %d has different size than kernel 0") % ii).str());
        }
        if ((ctrX != kernelList[ii]->getCtrX()) || (ctrY != kernelList[ii]->getCtrY())) {
            throw LSST_EXCEPT(ex::InvalidParameterException, 
                (boost::format("kernel %d has different center than kernel 0") % ii).str());
        }
        if (kernelList[ii]->isSpatiallyVarying()) {
            throw LSST_EXCEPT(ex::InvalidParameterException,
                (boost::format("kernel %d is spatially varying") % ii).str());
        }
    }
}

std::string lsst::afw::math::LinearCombinationKernel::toString(std::string prefix) const {
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
};


std::vector<double> lsst::afw::math::LinearCombinationKernel::getKernelParameters() const {
    return _kernelParams;
}

//
// Protected Member Functions
//
void lsst::afw::math::LinearCombinationKernel::setKernelParameter(unsigned int ind, double value) const {
    this->_kernelParams[ind] = value;
}

//
// Private Member Functions
//

/**
 * Compute _kernelImagePtrList, the internal archive of kernel images, and _kernelSumList, the sum of each kernel image.
 */
void lsst::afw::math::LinearCombinationKernel::_computeKernelImageList() {
    for (KernelList::const_iterator kIter = _kernelList.begin(), kEnd = _kernelList.end(); kIter != kEnd; ++kIter) {
        lsst::afw::image::Image<PixelT>::Ptr kernelImagePtr(new lsst::afw::image::Image<PixelT>(this->getDimensions()));
        _kernelSumList.push_back((*kIter)->computeImage(*kernelImagePtr, false));
        _kernelImagePtrList.push_back(kernelImagePtr);
    }
}
