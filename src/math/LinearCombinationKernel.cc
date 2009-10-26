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
    _kernelParams()
{ }

/**
 * @brief Construct a spatially invariant LinearCombinationKernel
 */
afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to const) kernels
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
afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to const) kernels
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
 * @brief Construct a spatially varying LinearCombinationKernel
 *  with the spatially varying parameters specified
 *
 * @throw lsst::pex::exceptions::InvalidParameterException  if the length of spatialFunctionList != # kernels
 */
afwMath::LinearCombinationKernel::LinearCombinationKernel(
    KernelList const &kernelList,    ///< list of (shared pointers to const) kernels
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)
        ///< list of spatial functions, one per kernel
:
    Kernel(kernelList[0]->getWidth(), kernelList[0]->getHeight(), spatialFunctionList),
    _kernelList(kernelList),
    _kernelImagePtrList(),
    _kernelSumList(),
    _kernelParams(std::vector<double>(kernelList.size()))
{
    if (kernelList.size() != spatialFunctionList.size()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
            "Length of spatialFunctionList does not match length of kernelList");
    }
    checkKernelList(kernelList);
    _computeKernelImageList();
}

afwMath::Kernel::Ptr afwMath::LinearCombinationKernel::clone() const {
    afwMath::Kernel::Ptr retPtr;
    if (this->isSpatiallyVarying()) {
        retPtr.reset(new afwMath::LinearCombinationKernel(this->_kernelList, this->_kernelParams));
    } else {
        retPtr.reset(new afwMath::LinearCombinationKernel(this->_kernelList, this->_spatialFunctionList));
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
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,"image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->computeKernelParametersFromSpatialModel(this->_kernelParams, x, y);
    }

    image = 0.0;
    double imSum = 0.0;
    std::vector<afwImage::Image<Pixel>::Ptr>::const_iterator kImPtrIter = _kernelImagePtrList.begin();
    std::vector<double>::const_iterator kSumIter = _kernelSumList.begin();
    std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    afwImage::Image<Pixel>::Ptr tmpImage(new afwImage::Image<Pixel>(image.getDimensions()));
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
 
std::string afwMath::LinearCombinationKernel::toString(std::string prefix) const {
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
 * Compute _kernelImagePtrList, the internal archive of kernel images,
 *  and _kernelSumList, the sum of each kernel image.
 */
void afwMath::LinearCombinationKernel::_computeKernelImageList() {
    for (KernelList::const_iterator kIter = _kernelList.begin(), kEnd = _kernelList.end();
        kIter != kEnd; ++kIter) {
        afwImage::Image<Pixel>::Ptr kernelImagePtr(new afwImage::Image<Pixel>(this->getDimensions()));
        _kernelSumList.push_back((*kIter)->computeImage(*kernelImagePtr, false));
        _kernelImagePtrList.push_back(kernelImagePtr);
    }
}

/**
 *  Return a ConvolutionVisitor that matches the type requested,at the given 
 *  location.
 *
 *  The default implementation would support the creation of IMAGE and FOURIER 
 *  visitors without derivatives. LinearCombinationKernel (and possibly 
 *  AnalyticKernel) can override to provide versions with derivatives.  
 */
lsst::afw::math::ImageConvolutionVisitor::Ptr 
lsst::afw::math::LinearCombinationKernel::computeImageConvolutionVisitor(
        lsst::afw::image::PointD const & location
) const{
    std::pair<int, int> center = std::make_pair(getCtrX(), getCtrY());
    lsst::afw::image::Image<Pixel>::Ptr imagePtr = 
            boost::make_shared<lsst::afw::image::Image<Pixel> >(getWidth(), getHeight());
    computeImage(*imagePtr, false, location.getX(), location.getY());
    std::vector<double> kernelParameters(getNKernelParameters());
    computeKernelParametersFromSpatialModel(kernelParameters, location.getX(), location.getY());
    return boost::make_shared<ImageConvolutionVisitor>(
            center, 
            kernelParameters, 
            imagePtr, 
            _kernelImagePtrList
    );
}

