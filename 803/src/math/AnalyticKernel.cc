// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definitions of AnalyticKernel member functions.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace ex = lsst::pex::exceptions;

/**
 * @brief Construct an empty spatially invariant AnalyticKernel of size 0x0
 */
lsst::afw::math::AnalyticKernel::AnalyticKernel()
:
    Kernel(),
    _kernelFunctionPtr()
{}

/**
 * @brief Construct a spatially invariant AnalyticKernel
 */
lsst::afw::math::AnalyticKernel::AnalyticKernel(
    int width,
    int height,
    KernelFunction const &kernelFunction)
:
    Kernel(width, height, kernelFunction.getNParameters()),
    _kernelFunctionPtr(kernelFunction.copy())
{}

/**
 * @brief Construct a spatially varying AnalyticKernel, replicating a spatial function once per kernel function parameter
 */
lsst::afw::math::AnalyticKernel::AnalyticKernel(
    int width,
    int height,
    KernelFunction const &kernelFunction,
    Kernel::SpatialFunction const &spatialFunction)
:
    Kernel(width, height, kernelFunction.getNParameters(), spatialFunction),
    _kernelFunctionPtr(kernelFunction.copy())
{}

/**
 * @brief Construct a spatially varying AnalyticKernel
 *
 * @throw lsst::pex::exceptions::InvalidParameterException
 *        if the length of spatialFunctionList != # kernel function parameters.
 */
lsst::afw::math::AnalyticKernel::AnalyticKernel(
    int width,
    int height,
    KernelFunction const &kernelFunction,
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)
:
    Kernel(width, height, spatialFunctionList),
    _kernelFunctionPtr(kernelFunction.copy())
{
    if (kernelFunction.getNParameters() != spatialFunctionList.size()) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
            "Length of spatialFunctionList does not match # of kernel function params");
    }
}

double lsst::afw::math::AnalyticKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    bool doNormalize,
    double x,
    double y
) const {
    typedef lsst::afw::image::Image<PixelT>::x_iterator x_iterator;
    
    if (image.getDimensions() != this->getDimensions()) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }

    double xOffset = -this->getCtrX();
    double yOffset = -this->getCtrY();

    double imSum = 0;
    for (int y = 0; y != this->getHeight(); ++y) {
        double const fy = y + yOffset;
        lsst::afw::image::Image<PixelT>::x_iterator ptr = image.row_begin(y);
        for (int x = 0; x != this->getWidth(); ++x, ++ptr) {
            double const fx = x + xOffset;
            PixelT const pixelVal = (*_kernelFunctionPtr)(fx, fy);
            *ptr = pixelVal;
            imSum += pixelVal;
        }
    }
    if (doNormalize) {
        image /= imSum;
        imSum = 1;
    }

    return imSum;
}

/**
 * @brief Get a deep copy of the kernel function
 */
lsst::afw::math::AnalyticKernel::KernelFunctionPtr lsst::afw::math::AnalyticKernel::getKernelFunction(
) const {
    return _kernelFunctionPtr->copy();
}

std::string lsst::afw::math::AnalyticKernel::toString(std::string prefix) const {
    std::ostringstream os;
    os << prefix << "AnalyticKernel:" << std::endl;
    os << prefix << "..function: " << (_kernelFunctionPtr ? _kernelFunctionPtr->toString() : "None") << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
};

std::vector<double> lsst::afw::math::AnalyticKernel::getKernelParameters() const {
    return _kernelFunctionPtr->getParameters();
}

//
// Protected Member Functions
//
void lsst::afw::math::AnalyticKernel::setKernelParameter(unsigned int ind, double value) const {
    _kernelFunctionPtr->setParameter(ind, value);
}
