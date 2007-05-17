// -*- LSST-C++ -*-
#include <stdexcept>

#include <vw/Image.h>

// #include <lsst/fw/Kernel.h>

namespace lsst {
namespace fw {

    template<typename PixelT>
    AnalyticKernel<PixelT>::AnalyticKernel(
    ) :
        Kernel<PixelT>(0),
        _kernelFunctionPtr()
    {}

    template<typename PixelT>
    AnalyticKernel<PixelT>::AnalyticKernel(
        typename Kernel<PixelT>::Function2PtrType kernelFunction,
        unsigned cols,
        unsigned rows
    ) :
        Kernel<PixelT>(cols, rows, kernelFunction->getNParameters()),
        _kernelFunctionPtr(kernelFunction)
    {}
    
    template<typename PixelT>
    AnalyticKernel<PixelT>::AnalyticKernel(
        typename Kernel<PixelT>::Function2PtrType kernelFunction,
        unsigned cols,
        unsigned rows,
        typename Kernel<PixelT>::Function2PtrType spatialFunction
    ) :
        Kernel<PixelT>(cols, rows, kernelFunction->getNParameters(), spatialFunction),
        _kernelFunctionPtr(kernelFunction)
    {}
    
    template<typename PixelT>
    AnalyticKernel<PixelT>::AnalyticKernel(
        typename Kernel<PixelT>::Function2PtrType kernelFunction,
        unsigned cols,
        unsigned rows,
        typename Kernel<PixelT>::Function2PtrType spatialFunction,
        const std::vector<std::vector<PixelT> > spatialParameters
    ) :
        Kernel<PixelT>(cols, rows, kernelFunction->getNParameters(), spatialFunction, spatialParameters),
        _kernelFunctionPtr(kernelFunction)
    {}

    template<typename PixelT>
    const Image<PixelT>
    AnalyticKernel<PixelT>::getImage(PixelT x, PixelT y) {
        if (this->isSpatiallyVarying()) {
            setKernelParametersFromSpatialModel(x, y);
        }
        Image<PixelT>image(this->getCols(), this->getRows());
        vw::MemoryStridingPixelAccessor<PixelT> imRow = image.getIVwPtr()->origin();
        int yBeg = - this->getCtrRow();
        int yEnd = this->getRows() - this->getCtrRow();
        int xBeg = - this->getCtrCol();
        int xEnd = this->getCols() - this->getCtrCol();
        for (int y = yBeg; y < yEnd; ++y) {
            vw::MemoryStridingPixelAccessor<PixelT> imCol = imRow;
            for (int x = xBeg; x < xEnd; ++x) {
                *imCol = (*_kernelFunctionPtr)(static_cast<PixelT>(x), static_cast<PixelT>(y));
                imCol.next_col();
            }
            imRow.next_row();
        }
        return image;
    }

    template<typename PixelT>
    inline typename Kernel<PixelT>::Function2PtrType
    AnalyticKernel<PixelT>::getKernelFunction() const {
        return _kernelFunctionPtr;
    }

    template<typename PixelT>
    inline void
    AnalyticKernel<PixelT>::setKernelParameters(const std::vector<PixelT> params) {
        _kernelFunctionPtr->setParameters(params);   
    }

    template<typename PixelT>
    inline const std::vector<PixelT>
    AnalyticKernel<PixelT>::getCurrentKernelParameters() const {
        return _kernelFunctionPtr->getParameters();
    }

}   // namespace fw
}   // namespace lsst
