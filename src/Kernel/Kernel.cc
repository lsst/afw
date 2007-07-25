// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of Kernel member functions and explicit instantiations of the class.
 *
 * \ingroup fw
 */
#include <stdexcept>

#include <boost/format.hpp>

#include <lsst/fw/Kernel.h>

//
// Constructors
//

/**
 * \brief Construct a spatially invariant Kernel with no kernel parameters
 */
template<typename PixelT>
lsst::fw::Kernel<PixelT>::Kernel()
:
    LsstBase(typeid(this)),
   _cols(0),
   _rows(0),
   _ctrCol(0),
   _ctrRow(0),
   _nKernelParams(0),
   _isSpatiallyVarying(false),
   _spatialFunctionPtr()
{ }

/**
 * \brief Construct a spatially invariant Kernel
 */
template<typename PixelT>
lsst::fw::Kernel<PixelT>::Kernel(
    unsigned int cols,  ///< number of columns
    unsigned int rows,  ///< number of rows
    unsigned int nKernelParams) ///< number of kernel parameters
:
    LsstBase(typeid(this)),
   _cols(cols),
   _rows(rows),
   _ctrCol((cols-1)/2),
   _ctrRow((rows-1)/2),
   _nKernelParams(nKernelParams),
   _isSpatiallyVarying(false),
   _spatialFunctionPtr()
{ }

/**
 * \brief Construct a spatially varying Kernel with spatial parameters initialized to 0
 *
 * \throw std::invalid_argument if the kernel or spatial function has no parameters.
 */
template<typename PixelT>
lsst::fw::Kernel<PixelT>::Kernel(
    unsigned int cols,  ///< number of columns
    unsigned int rows,  ///< number of rows
    unsigned int nKernelParams, ///< number of kernel parameters
    SpatialFunctionPtrType spatialFunction) ///< spatial function
:
    LsstBase(typeid(this)),
   _cols(cols),
   _rows(rows),
   _ctrCol((cols-1)/2),
   _ctrRow((rows-1)/2),
   _nKernelParams(nKernelParams),
   _isSpatiallyVarying(true),
   _spatialFunctionPtr(spatialFunction)
{
    // create spatial parameters initialized to zero
    if (this->getNSpatialParameters() == 0) {
        throw std::invalid_argument("Spatial function has no parameters");
    }
    if (this->getNKernelParameters() == 0) {
        throw std::invalid_argument("Kernel function has no parameters");
    }
    std::vector<double> zeros(this->getNSpatialParameters());
    std::vector<std::vector<double> > spatialParams;
    for (unsigned int ii = 0; ii < this->getNKernelParameters(); ++ii) {
        this->_spatialParams.push_back(zeros);
    }
}

/**
 * \brief Construct a spatially varying Kernel with the spatially varying parameters specified
 *
 * See setSpatialParameters for the form of the spatial parameters.
 *
 * \throw std::invalid_argument if:
 * - the kernel or spatial function has no parameters
 * - the spatialParams vector is the wrong length
 */
template<typename PixelT>
lsst::fw::Kernel<PixelT>::Kernel(
    unsigned int cols,  ///< number of columns
    unsigned int rows,  ///< number of rows
    unsigned int nKernelParams, ///< number of kernel parameters
    SpatialFunctionPtrType spatialFunction, ///< spatial function
    std::vector<std::vector<double> > const &spatialParams)  ///< spatial parameters
:
    LsstBase(typeid(this)),
   _cols(cols),
   _rows(rows),
   _ctrCol((cols-1)/2),
   _ctrRow((rows-1)/2),
   _nKernelParams(nKernelParams),
   _isSpatiallyVarying(true),
   _spatialFunctionPtr(spatialFunction){
    if (this->getNSpatialParameters() == 0) {
        throw std::invalid_argument("Spatial function has no parameters");
    }
    if (this->getNKernelParameters() == 0) {
        throw std::invalid_argument("Kernel function has no parameters");
    }
    setSpatialParameters(spatialParams);
}

//
// Public Member Functions
//

/**
 * \brief Compute an image (pixellized representation of the kernel), returning a new image
 *
 * This would be called computeImage (overloading the other function of the same name)
 * but at least some versions of the g++ compiler cannot then reliably find the function.
 *
 * \return an image (your own copy to do with as you wish)
 */
template<typename PixelT>
lsst::fw::Image<PixelT> lsst::fw::Kernel<PixelT>::computeNewImage(
    PixelT &imSum,  ///< sum of image pixels
    double x,       ///< x (column position) at which to compute spatial function
    double y,       ///< y (row position) at which to compute spatial function
    bool doNormalize    ///< normalize the image (so sum is 1)?
) const {
    lsst::fw::Image<PixelT> retImage(this->getCols(), this->getRows());
    this->computeImage(retImage, imSum, x, y, doNormalize);
    return retImage;
}

/**
 * \brief Return the kernel parameters at a specified position
 *
 * If the kernel is not spatially varying then the position is ignored
 * If there are no kernel parameters then an empty vector is returned
 */
template<typename PixelT>
std::vector<double> lsst::fw::Kernel<PixelT>::getKernelParameters(
    double x,   ///< x at which to evaluate the spatial model
    double y    ///< y at which to evaluate the spatial model
) const {
   if (isSpatiallyVarying()) {
       return getKernelParametersFromSpatialModel(x, y);
   } else {
       return getCurrentKernelParameters();
   }
}

/**
 * \brief Set the parameters of the spatial function for all kernel parameters
 *
 * Params is indexed as [kernel parameter][spatial parameter]
 *
 * \throw std::invalid_argument if params is the wrong shape.
 */
template<typename PixelT>
void lsst::fw::Kernel<PixelT>::setSpatialParameters(const std::vector<std::vector<double> > params) {
    unsigned int nKernelParams = this->getNKernelParameters();
    if (params.size() != nKernelParams) {
        throw std::invalid_argument(str(
            boost::format("params has %d entries instead of %d") % params.size() % nKernelParams
        ));
    }
    unsigned int nSpatialParams = this->getNSpatialParameters();
    for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
        if (params[ii].size() != nSpatialParams) {
            throw std::invalid_argument(str(
                boost::format("params[%d] has %d entries instead of %d") %
                ii % params[ii].size() % nSpatialParams
            ));
        }
    }
    _spatialParams = params;
}

//
// Protected Member Functions
//

/**
 * \brief Return the kernel parameters at a specified point
 *
 * Assumes there is a spatial model
 */
template<typename PixelT>
std::vector<double> lsst::fw::Kernel<PixelT>::getKernelParametersFromSpatialModel(double x, double y) const {
    std::vector<double> kernelParams(getNKernelParameters());
    typename std::vector<double>::iterator kIter;
    typename std::vector<std::vector<double> >::const_iterator sIter;
    for (kIter = kernelParams.begin(), sIter = _spatialParams.begin();
        kIter != kernelParams.end(); ++kIter, ++sIter) {
        _spatialFunctionPtr->setParameters(*sIter);
        *kIter = (*_spatialFunctionPtr)(x,y);
    }
    return kernelParams;
}
   
/**
 * \brief Set the kernel parameters at a specified point
 *
 * Assumes there is a spatial model
 */
template<typename PixelT>
void lsst::fw::Kernel<PixelT>::setKernelParametersFromSpatialModel(double x, double y) const {
    this->basicSetKernelParameters(getKernelParametersFromSpatialModel(x,y));
}

// Explicit instantiations
template class lsst::fw::Kernel<float>;
template class lsst::fw::Kernel<double>;
