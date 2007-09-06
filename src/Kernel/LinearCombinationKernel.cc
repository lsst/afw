// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of LinearCombinationKernel member functions and explicit instantiations of the class.
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <stdexcept>

#include <boost/format.hpp>
#include <vw/Image.h>

#include <lsst/mwi/exceptions/Exception.h>
#include <lsst/fw/Kernel.h>

// This file is meant to be included by lsst/fw/Kernel.h

/**
 * \brief Construct an empty LinearCombinationKernel of size 0x0
 */
template<typename PixelT>
lsst::fw::LinearCombinationKernel<PixelT>::LinearCombinationKernel()
:
    Kernel<PixelT>(),
    _kernelList(),
    _kernelImageList(),
    _kernelParams()
{ }

/**
 * \brief Construct a spatially invariant LinearCombinationKernel
 */
template<typename PixelT>
lsst::fw::LinearCombinationKernel<PixelT>::LinearCombinationKernel(
    KernelListType const &kernelList,    ///< list of (shared pointers to) kernels
    std::vector<double> const &kernelParameters) ///< kernel coefficients
:
    Kernel<PixelT>(kernelList[0]->getCols(), kernelList[0]->getRows(), kernelList.size()),
    _kernelList(kernelList),
    _kernelImageList(),
    _kernelParams(kernelParameters)
{
    checkKernelList(kernelList);
    _computeKernelImageList();
}

/**
 * \brief Construct a spatially varying LinearCombinationKernel with spatial parameters initialized to 0
 */
template<typename PixelT>
lsst::fw::LinearCombinationKernel<PixelT>::LinearCombinationKernel(
    KernelListType const &kernelList,    ///< list of (shared pointers to) kernels
    typename Kernel<PixelT>::SpatialFunctionPtrType spatialFunction)  ///< spatial function
:
    Kernel<PixelT>(kernelList[0]->getCols(), kernelList[0]->getRows(), kernelList.size(), spatialFunction),
    _kernelList(kernelList),
    _kernelImageList(),
    _kernelParams(std::vector<double>(kernelList.size()))
{
    checkKernelList(kernelList);
    _computeKernelImageList();
}

/**
 * \brief Construct a spatially varying LinearCombinationKernel with the spatially varying parameters specified
 *
 * See setSpatialParameters for the form of the spatial parameters.
 */
template<typename PixelT>
lsst::fw::LinearCombinationKernel<PixelT>::LinearCombinationKernel(
    KernelListType const &kernelList,    ///< list of (shared pointers to) kernels
    typename Kernel<PixelT>::SpatialFunctionPtrType spatialFunction,  ///< spatial function
    std::vector<std::vector<double> > const &spatialParameters)  ///< spatial coefficients
:
    Kernel<PixelT>(kernelList[0]->getCols(), kernelList[0]->getRows(),
        kernelList.size(), spatialFunction, spatialParameters),
    _kernelList(kernelList),
    _kernelImageList(),
    _kernelParams(std::vector<double>(kernelList.size()))
{
    checkKernelList(kernelList);
    _computeKernelImageList();
}

template<typename PixelT>
void lsst::fw::LinearCombinationKernel<PixelT>::computeImage(
    Image<PixelT> &image,
    PixelT &imSum,
    double x,
    double y,
    bool doNormalize
) const {
    if ((image.getCols() != this->getCols()) || (image.getRows() != this->getRows())) {
        throw lsst::mwi::exceptions::InvalidParameter("image is the wrong size");
    }
    if (this->isSpatiallyVarying()) {
        this->setKernelParametersFromSpatialModel(x, y);
    }
    Image<PixelT> tempImage(this->getCols(), this->getRows());
    typename Image<PixelT>::ImageIVwPtrT vwImagePtr = image.getIVwPtr();
    
    typename std::vector<Image<PixelT> >::const_iterator kImIter = _kernelImageList.begin();
    typename std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    for ( ; kImIter != _kernelImageList.end(); ++kImIter, ++kParIter) {
        *vwImagePtr += (*(kImIter->getIVwPtr())) * static_cast<PixelT>(*kParIter);
    }
    imSum = vw::sum_of_channel_values(*vwImagePtr);
    if (doNormalize) {
        image /= imSum;
        imSum = 1;
    }
}

/**
 * \brief Get the fixed basis kernels
 */
template<typename PixelT>
typename lsst::fw::LinearCombinationKernel<PixelT>::KernelListType const &
lsst::fw::LinearCombinationKernel<PixelT>::getKernelList() const {
    return _kernelList;
}
    
/**
 * \brief Check that all kernels have the same size and center and that none are spatially varying
 *
 * \throw lsst::mwi::exceptions::InvalidParameter if the check fails
 */
template<typename PixelT>
void lsst::fw::LinearCombinationKernel<PixelT>::checkKernelList(const KernelListType &kernelList) const {
    if (kernelList.size() < 1) {
        throw lsst::mwi::exceptions::InvalidParameter("kernelList has no elements");
    }

    unsigned int cols = kernelList[0]->getCols();
    unsigned int rows = kernelList[0]->getRows();
    unsigned int ctrCol = kernelList[0]->getCtrCol();
    unsigned int ctrRow = kernelList[0]->getCtrRow();
    
    for (unsigned int ii = 0; ii < kernelList.size(); ii++) {
        if (ii > 0) {
            if ((cols != kernelList[ii]->getCols()) ||
                (rows != kernelList[ii]->getRows())) {
                throw lsst::mwi::exceptions::InvalidParameter(
                    boost::format("kernel %d has different size than kernel 0") % ii);
            }
            if ((ctrCol != kernelList[ii]->getCtrCol()) ||
                (ctrRow != kernelList[ii]->getCtrRow())) {
                throw lsst::mwi::exceptions::InvalidParameter(
                    boost::format("kernel %d has different center than kernel 0") % ii);
            }
        }
        if (kernelList[ii]->isSpatiallyVarying()) {
            throw lsst::mwi::exceptions::InvalidParameter(
                boost::format("kernel %d is spatially varying") % ii);
        }
    }
}

template<typename PixelT>
std::vector<double> lsst::fw::LinearCombinationKernel<PixelT>::getCurrentKernelParameters() const {
    return _kernelParams;
}

//
// Protected Member Functions
//

template<typename PixelT>
void lsst::fw::LinearCombinationKernel<PixelT>::basicSetKernelParameters(std::vector<double> const &params)
const {
    if (params.size() != this->_kernelList.size()) {
        throw lsst::mwi::exceptions::InvalidParameter(
            boost::format("params size is %d instead of %d\n") % params.size() % _kernelList.size());
    }
    this->_kernelParams = params;
}

//
// Private Member Functions
//

/**
 * Compute _kernelImageList, the internal archive of kernel images.
 */
template<typename PixelT>
void lsst::fw::LinearCombinationKernel<PixelT>::_computeKernelImageList() {
    typename KernelListType::const_iterator kIter = _kernelList.begin();
    typename std::vector<double>::const_iterator kParIter = _kernelParams.begin();
    for ( ; kIter != _kernelList.end(); ++kIter) {
        PixelT kSum;
        _kernelImageList.push_back((*kIter)->computeNewImage(kSum, 0, 0, false));
    }
}

// Explicit instantiations
template class lsst::fw::LinearCombinationKernel<float>;
template class lsst::fw::LinearCombinationKernel<double>;
