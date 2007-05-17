// -*- LSST-C++ -*-
/**
 * To do: add spatial model support to getImage.
 */

#include <stdexcept>

#include <boost/format.hpp>

// #include <lsst/fw/Kernel.h>

namespace lsst {
namespace fw {

    template<typename PixelT>
    LinearCombinationKernel<PixelT>::LinearCombinationKernel(
    ) :
        Kernel<PixelT>(),
        _kernelList(),
        _kernelParams()
    {
    }

    template<typename PixelT>
    LinearCombinationKernel<PixelT>::LinearCombinationKernel(
        const KernelListType kernelList,
        const std::vector<PixelT> kernelParameters
    ) :
        Kernel<PixelT>(kernelList[0]->getCols(), kernelList[0]->getRows(), kernelList.size()),
        _kernelList(kernelList),
        _kernelParams(kernelParameters)
    {
        checkKernelList(kernelList);
    }
    
    template<typename PixelT>
    LinearCombinationKernel<PixelT>::LinearCombinationKernel(
        const KernelListType kernelList,
        typename Kernel<PixelT>::Function2PtrType spatialFunction
    ) :
        Kernel<PixelT>(kernelList[0]->getCols(), kernelList[0]->getRows(), kernelList.size(), spatialFunction),
        _kernelList(kernelList),
        _kernelParams(std::vector<PixelT>(kernelList.size()))
    {
        checkKernelList(kernelList);
    }
        
    template<typename PixelT>
    LinearCombinationKernel<PixelT>::LinearCombinationKernel(
        const KernelListType kernelList,
        typename Kernel<PixelT>::Function2PtrType spatialFunction,
        const std::vector<std::vector<PixelT> > spatialParameters
    ) :
        Kernel<PixelT>(kernelList[0]->getCols(), kernelList[0]->getRows(),
            kernelList.size(), spatialFunction, spatialParameters),
        _kernelList(kernelList),
        _kernelParams(std::vector<PixelT>(kernelList.size()))
    {
        checkKernelList(kernelList);
    }
    
    template<typename PixelT>
    const Image<PixelT>
    LinearCombinationKernel<PixelT>::getImage(PixelT x, PixelT y) {
        Image<PixelT> image(this->getCols(), this->getRows());
        typename Image<PixelT>::ImageIVwPtrT vwImagePtr = image.getVwPtr();
        
        if (this->isSpatiallyVarying()) {
            throw std::invalid_argument("LinearCombinationKernel.getImage can't yet handle a spatial model");
        } else {
            typename KernelListType::const_iterator kIter = _kernelList.begin();
            typename std::vector<double>::const_iterator kParIter = _kernelParams.begin();
            
            for ( ; kIter != _kernelList.end(); ++kIter, ++kParIter) {
                **kIter;
                *vwImagePtr += (*((*kIter)->getImage().getVwPtr())) * (*kParIter);
            }
        }
        return image;
    }

    template<typename PixelT>
    inline const typename LinearCombinationKernel<PixelT>::KernelListType &
    LinearCombinationKernel<PixelT>::getKernelList() const {
        return _kernelList;
    }
        
    template<typename PixelT>
    void
    LinearCombinationKernel<PixelT>::checkKernelList(const KernelListType &kernelList) const {
        if (kernelList.size() < 1) {
            throw std::invalid_argument("kernelList has no elements");
        }

        unsigned cols = kernelList[0]->getCols();
        unsigned rows = kernelList[0]->getRows();
        unsigned ctrCol = kernelList[0]->getCtrCol();
        unsigned ctrRow = kernelList[0]->getCtrRow();
        
        for (unsigned ii = 0; ii < kernelList.size(); ii++) {
            if (ii > 0) {
                if ((cols != kernelList[ii]->getCols()) ||
                    (rows != kernelList[ii]->getRows())) {
                    throw std::invalid_argument(str(
                        boost::format("kernel %d has different size than kernel 0") % ii
                    ));
                }
                if ((ctrCol != kernelList[ii]->getCtrCol()) ||
                    (ctrRow != kernelList[ii]->getCtrRow())) {
                    throw std::invalid_argument(str(
                        boost::format("kernel %d has different center than kernel 0") % ii
                    ));
                }
            }
            if (kernelList[ii]->isSpatiallyVarying()) {
                throw std::invalid_argument(str(
                    boost::format("kernel %d is spatially varying") % ii
                ));
            }
        }
    }
    
    template<typename PixelT>
    inline void
    LinearCombinationKernel<PixelT>::setKernelParameters(const std::vector<PixelT> params) {
        if (params.size() != this->_kernelList.size()) {
            std::ostringstream errStream;
            errStream << "setKernelParameters called with " << params.size() << 
                " parameters instead of " << _kernelList.size() << std::endl;
            throw std::invalid_argument(errStream.str());            
        }
        this->_kernelParams = params;
    }

    template<typename PixelT>
    inline const std::vector<PixelT>
    LinearCombinationKernel<PixelT>::getCurrentKernelParameters() const {
        return _kernelParams;
    }

}   // namespace fw
}   // namespace lsst
