// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of AnalyticKernel member functions and explicit instantiations of the class.
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <vw/Image.h>

#include <lsst/fw/Kernel.h>

// This file is meant to be included by lsst/fw/Kernel.h

/**
 * \brief Construct an empty spatially invariant AnalyticKernel of size 0x0
 */
template<typename PixelT>
lsst::fw::AnalyticKernel<PixelT>::AnalyticKernel()
:
    Kernel<PixelT>(),
    _kernelFunctionPtr()
{}

/**
 * \brief Construct a spatially invariant AnalyticKernel
 */
template<typename PixelT>
lsst::fw::AnalyticKernel<PixelT>::AnalyticKernel(
    typename Kernel<PixelT>::KernelFunctionPtrType kernelFunction,
    unsigned int cols,
    unsigned int rows)
:
    Kernel<PixelT>(cols, rows, kernelFunction->getNParameters()),
    _kernelFunctionPtr(kernelFunction)
{}

/**
 * \brief Construct a spatially varying AnalyticKernel with spatial coefficients initialized to 0
 */
template<typename PixelT>
lsst::fw::AnalyticKernel<PixelT>::AnalyticKernel(
    typename Kernel<PixelT>::KernelFunctionPtrType kernelFunction,
    unsigned int cols,
    unsigned int rows,
    typename Kernel<PixelT>::SpatialFunctionPtrType spatialFunction)
:
    Kernel<PixelT>(cols, rows, kernelFunction->getNParameters(), spatialFunction),
    _kernelFunctionPtr(kernelFunction)
{}

/**
 * \brief Construct a spatially varying AnalyticKernel with the spatially varying parameters specified
 *
 * See setSpatialParameters for the form of the spatial parameters.
 */
template<typename PixelT>
lsst::fw::AnalyticKernel<PixelT>::AnalyticKernel(
    typename Kernel<PixelT>::KernelFunctionPtrType kernelFunction,
    unsigned int cols,
    unsigned int rows,
    typename Kernel<PixelT>::SpatialFunctionPtrType spatialFunction,
    std::vector<std::vector<double> > const &spatialParameters)
:
    Kernel<PixelT>(cols, rows, kernelFunction->getNParameters(), spatialFunction, spatialParameters),
    _kernelFunctionPtr(kernelFunction)
{}

template<typename PixelT>
void lsst::fw::AnalyticKernel<PixelT>::computeImage(
    Image<PixelT> &image,
    PixelT &imSum,
    double x,
    double y,
    bool doNormalize
) const {
    typedef typename Image<PixelT>::pixel_accessor pixelAccessor;
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::mwi::exceptions::InvalidParameter("image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
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
template<typename PixelT>
typename lsst::fw::Kernel<PixelT>::KernelFunctionPtrType lsst::fw::AnalyticKernel<PixelT>::getKernelFunction(
) const {
    return _kernelFunctionPtr;
}

template<typename PixelT>
std::vector<double> lsst::fw::AnalyticKernel<PixelT>::getCurrentKernelParameters() const {
    return _kernelFunctionPtr->getParameters();
}

//
// Protected Member Functions
//

template<typename PixelT>
void lsst::fw::AnalyticKernel<PixelT>::basicSetKernelParameters(std::vector<double> const &params) const {
    _kernelFunctionPtr->setParameters(params);
}

// Explicit instantiations
template class lsst::fw::AnalyticKernel<float>;
template class lsst::fw::AnalyticKernel<double>;
