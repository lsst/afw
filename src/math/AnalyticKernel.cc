// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of AnalyticKernel member functions and explicit instantiations of the class.
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <vw/Image.h>

#include <lsst/pex/exceptions.h>
#include <lsst/afw/math/Kernel.h>

// This file is meant to be included by lsst/afw/math/Kernel.h

/**
 * \brief Construct an empty spatially invariant AnalyticKernel of size 0x0
 */
lsst::afw::math::AnalyticKernel::AnalyticKernel()
:
    Kernel(),
    _kernelFunctionPtr()
{}

/**
 * \brief Construct a spatially invariant AnalyticKernel
 */
lsst::afw::math::AnalyticKernel::AnalyticKernel(
    Kernel::KernelFunction const &kernelFunction,
    unsigned int cols,
    unsigned int rows)
:
    Kernel(cols, rows, kernelFunction.getNParameters()),
    _kernelFunctionPtr(kernelFunction.copy())
{}

/**
 * \brief Construct a spatially varying AnalyticKernel, replicating a spatial function once per kernel function parameter
 */
lsst::afw::math::AnalyticKernel::AnalyticKernel(
    Kernel::KernelFunction const &kernelFunction,
    unsigned int cols,
    unsigned int rows,
    Kernel::SpatialFunction const &spatialFunction)
:
    Kernel(cols, rows, kernelFunction.getNParameters(), spatialFunction),
    _kernelFunctionPtr(kernelFunction.copy())
{}

/**
 * \brief Construct a spatially varying AnalyticKernel
 *
 * \throw lsst::pex::exceptions::InvalidParameter if the length of spatialFunctionList != # kernel function parameters.
 */
lsst::afw::math::AnalyticKernel::AnalyticKernel(
    Kernel::KernelFunction const &kernelFunction,
    unsigned int cols,
    unsigned int rows,
    std::vector<Kernel::SpatialFunctionPtr> const &spatialFunctionList)
:
    Kernel(cols, rows, spatialFunctionList),
    _kernelFunctionPtr(kernelFunction.copy())
{
    if (kernelFunction.getNParameters() != spatialFunctionList.size()) {
        throw lsst::pex::exceptions::InvalidParameter("Length of spatialFunctionList does not match # of kernel function params");
    }
}

void lsst::afw::math::AnalyticKernel::computeImage(
    lsst::afw::image::Image<PixelT> &image,
    PixelT &imSum,
    double x,
    double y,
    bool doNormalize
) const {
    typedef lsst::afw::image::Image<PixelT>::pixel_accessor pixelAccessor;
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::pex::exceptions::InvalidParameter("image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        std::vector<double> kernelParams(this->getNKernelParameters());
        this->computeKernelParametersFromSpatialModel(kernelParams, x, y);
        this->basicSetKernelParameters(kernelParams);
    }
    pixelAccessor imRow = image.origin();
    double xOffset = - static_cast<double>(this->getCtrCol());
    double yOffset = - static_cast<double>(this->getCtrRow());
    imSum = 0;
    for (unsigned int row = 0; row < this->getRows(); ++row, imRow.next_row()) {
        double y = static_cast<double>(row) + yOffset;
        pixelAccessor imCol = imRow;
        for (unsigned int col = 0; col < this->getCols(); ++col, imCol.next_col()) {
            double x = static_cast<double>(col) + xOffset;
            PixelT pixelVal = (*_kernelFunctionPtr)(x, y);
            *imCol = pixelVal;
            imSum += pixelVal;
        }
    }
    if (doNormalize) {
        image /= imSum;
        imSum = 1;
    }
}

/**
 * \brief Get the kernel function
 */
lsst::afw::math::Kernel::KernelFunctionPtr lsst::afw::math::AnalyticKernel::getKernelFunction(
) const {
    return _kernelFunctionPtr;
}

std::vector<double> lsst::afw::math::AnalyticKernel::getCurrentKernelParameters() const {
    return _kernelFunctionPtr->getParameters();
}

//
// Protected Member Functions
//

void lsst::afw::math::AnalyticKernel::basicSetKernelParameters(std::vector<double> const &params) const {
    _kernelFunctionPtr->setParameters(params);
}
