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

/**
 * @brief Construct an empty LinearCombinationKernel of size 0x0
 */
lsst::afw::math::LinearCombinationKernel::LinearCombinationKernel()
:
    Kernel(),
    _kernelList(),
    _kernelImagePtrList(),
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
    _kernelParams(std::vector<double>(kernelList.size()))
{
    checkKernelList(kernelList);
    _computeKernelImageList();
}

/**
 * @brief Construct a spatially varying LinearCombinationKernel with the spatially varying parameters specified
 *
 * @throw lsst::pex::exceptions::InvalidParameter if the length of spatialFunctionList != # kernels
 */
lsst::afw::math::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to) kernels
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)    ///< list of spatial functions, one per kernel
:
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), spatialFunctionList),
    _kernelList(kernelList),
    _kernelImagePtrList(),
    _kernelParams(std::vector<double>(kernelList.size()))
{
    if (kernelList.size() != spatialFunctionList.size()) {
        throw lsst::pex::exceptions::InvalidParameter("Length of spatialFunctionList does not match length of kernelList");
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
    if (image.dimensions() != this->dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->computeKernelParametersFromSpatialModel(this->_kernelParams, x, y);
    }
    
    std::vector<lsst::afw::image::Image<PixelT>::Ptr>::const_iterator kImPtrIter = _kernelImagePtrList.begin();
    std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    //
    // Temp image to generate the kernel*components into.  Image::operator*() doesn't exist
    // as it'd have to generate a temporary, and I don't think it's a good idea to make tmps
    // without explicit requests from the user.
    //
    lsst::afw::image::Image<PixelT>::Ptr tmpImage(new lsst::afw::image::Image<PixelT>(image.dimensions()));

    image <<= **kImPtrIter;
    image *= *kParIter;
    ++kImPtrIter, ++kParIter;
    
    for ( ; kImPtrIter != _kernelImagePtrList.end(); ++kImPtrIter, ++kParIter) {
        *tmpImage <<= **kImPtrIter;
        *tmpImage *= *kParIter;
        image += *tmpImage;
    }

    double imSum = 0;
    imSum = std::accumulate(image.begin(), image.end(), imSum);

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
 * @throw lsst::pex::exceptions::InvalidParameter if the check fails
 */
void lsst::afw::math::LinearCombinationKernel::checkKernelList(const KernelList &kernelList) const {
    if (kernelList.size() < 1) {
        throw lsst::pex::exceptions::InvalidParameter("kernelList has no elements");
    }
    
    int ctrX = kernelList[0]->getCtrX();
    int ctrY = kernelList[0]->getCtrY();
    
    for (unsigned int ii = 0; ii < kernelList.size(); ++ii) {
        if (kernelList[ii]->dimensions() != kernelList[0]->dimensions()) {
            throw lsst::pex::exceptions::InvalidParameter(
                                                  boost::format("kernel %d has different size than kernel 0") % ii);
        }
        if ((ctrX != kernelList[ii]->getCtrX()) || (ctrY != kernelList[ii]->getCtrY())) {
            throw lsst::pex::exceptions::InvalidParameter(
                                                  boost::format("kernel %d has different center than kernel 0") % ii);
        }
        if (kernelList[ii]->isSpatiallyVarying()) {
            throw lsst::pex::exceptions::InvalidParameter(
                boost::format("kernel %d is spatially varying") % ii);
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
 * Compute _kernelImagePtrList, the internal archive of kernel images.
 */
void lsst::afw::math::LinearCombinationKernel::_computeKernelImageList() {
    std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    for (KernelList::const_iterator kIter = _kernelList.begin(), kEnd = _kernelList.end(); kIter != kEnd; ++kIter) {
        lsst::afw::image::Image<PixelT>::Ptr kernelImagePtr(new lsst::afw::image::Image<PixelT>(this->dimensions()));

        (void)(*kIter)->computeImage(*kernelImagePtr, false);

        _kernelImagePtrList.push_back(kernelImagePtr);
    }
}
