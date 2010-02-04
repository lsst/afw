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
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
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
    _kernelParams()
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
    _kernelParams(kernelParameters)
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
    _kernelParams(std::vector<double>(kernelList.size()))
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
    _kernelParams(std::vector<double>(kernelList.size()))
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
    afwMath::Kernel::Ptr retPtr;
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
    afwImage::Image<Pixel>::Ptr tmpImage(new afwImage::Image<Pixel>(image.getDimensions()));
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
 * @brief Set _kernelList by cloning each input kernel and update the kernel image cache.
 */
void afwMath::LinearCombinationKernel::_setKernelList(KernelList const &kernelList) {
    _kernelSumList.clear();
    _kernelImagePtrList.clear();
    _kernelList.clear();
    for (KernelList::const_iterator kIter = kernelList.begin(), kEnd = kernelList.end();
        kIter != kEnd; ++kIter) {
        Kernel::Ptr basisKernelPtr = (*kIter)->clone();
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
    lsst::afw::geom::Point2I center = lsst::afw::geom::makePointI(
        getCtrX(), getCtrY()
    );
    computeImage(*imagePtr, false, location.getX(), location.getY());
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

