// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definitions of Kernel member functions and explicit instantiations of the class.
 *
 * \ingroup afw
 */
#include <stdexcept>

#include <boost/format.hpp>

#include <lsst/pex/exceptions.h>
#include <lsst/afw/math/Kernel.h>

lsst::afw::math::generic_kernel_tag lsst::afw::math::generic_kernel_tag_; ///< Used as default value in argument lists
lsst::afw::math::deltafunction_kernel_tag lsst::afw::math::deltafunction_kernel_tag_; ///< Used as default value in argument lists

//
// Constructors
//

/**
 * \brief Construct a spatially invariant Kernel with no kernel parameters
 */
lsst::afw::math::Kernel::Kernel()
:
    LsstBase(typeid(this)),
   _cols(0),
   _rows(0),
   _ctrCol(0),
   _ctrRow(0),
   _nKernelParams(0),
   _spatialFunctionList()
{ }

/**
 * \brief Construct a spatially invariant Kernel
 */
lsst::afw::math::Kernel::Kernel(
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
   _spatialFunctionList()
{ }

/**
 * \brief Construct a spatially varying Kernel with one spatial function copied as needed
 *
 * \throw lsst::pex::exceptions::InvalidParameter if the kernel has no parameters.
 */
lsst::afw::math::Kernel::Kernel(
    unsigned int cols,  ///< number of columns
    unsigned int rows,  ///< number of rows
    unsigned int nKernelParams, ///< number of kernel parameters
    SpatialFunction const &spatialFunction) ///< spatial function
:
    LsstBase(typeid(this)),
   _cols(cols),
   _rows(rows),
   _ctrCol((cols-1)/2),
   _ctrRow((rows-1)/2),
   _nKernelParams(nKernelParams)
{
    if (nKernelParams == 0) {
        throw lsst::pex::exceptions::InvalidParameter("Kernel function has no parameters");
    }
    for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
        SpatialFunctionPtr spatialFunctionCopy = spatialFunction.copy();
        this->_spatialFunctionList.push_back(spatialFunctionCopy);
    }
}

/**
 * \brief Construct a spatially varying Kernel with a list of spatial functions (one per kernel parameter)
 *
 * Note: if the list of spatial functions is empty then the kernel is not spatially varying.
 */
lsst::afw::math::Kernel::Kernel(
    unsigned int cols,  ///< number of columns
    unsigned int rows,  ///< number of rows
    std::vector<SpatialFunctionPtr> spatialFunctionList) ///< list of spatial function, one per kernel parameter
:
    LsstBase(typeid(this)),
   _cols(cols),
   _rows(rows),
   _ctrCol((cols-1)/2),
   _ctrRow((rows-1)/2),
   _nKernelParams(spatialFunctionList.size())
{
    for (unsigned int ii = 0; ii < spatialFunctionList.size(); ++ii) {
        SpatialFunctionPtr spatialFunctionCopy = spatialFunctionList[ii]->copy();
        this->_spatialFunctionList.push_back(spatialFunctionCopy);
    }
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
lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT> lsst::afw::math::Kernel::computeNewImage(
    PixelT &imSum,  ///< sum of image pixels
    double x,       ///< x (column position) at which to compute spatial function
    double y,       ///< y (row position) at which to compute spatial function
    bool doNormalize    ///< normalize the image (so sum is 1)?
) const {
    lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT> retImage(this->getCols(), this->getRows());
    this->computeImage(retImage, imSum, x, y, doNormalize);
    return retImage;
}

/**
 * \brief Return the kernel parameters at a specified position
 *
 * If the kernel is not spatially varying then the position is ignored
 * If there are no kernel parameters then an empty vector is returned
 */
std::vector<double> lsst::afw::math::Kernel::getKernelParameters(
    double x,   ///< x at which to evaluate the spatial model
    double y    ///< y at which to evaluate the spatial model
) const {
    if (isSpatiallyVarying()) {
        std::vector<double> kernelParams(getNKernelParameters());
        computeKernelParametersFromSpatialModel(kernelParams, x, y);
        return kernelParams;
    } else {
        return getCurrentKernelParameters();
    }
}

/**
 * \brief Set the parameters of all spatial functions
 *
 * Params is indexed as [kernel parameter][spatial parameter]
 *
 * \throw lsst::pex::exceptions::InvalidParameter if params is the wrong shape (and no parameters are changed)
 */
void lsst::afw::math::Kernel::setSpatialParameters(const std::vector<std::vector<double> > params) {
    // Check params size before changing anything
    unsigned int nKernelParams = this->getNKernelParameters();
    if (params.size() != nKernelParams) {
        throw lsst::pex::exceptions::InvalidParameter(
            boost::format("params has %d entries instead of %d") % params.size() % nKernelParams);
    }
    unsigned int nSpatialParams = this->getNSpatialParameters();
    for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
        if (params[ii].size() != nSpatialParams) {
            throw lsst::pex::exceptions::InvalidParameter(
                boost::format("params[%d] has %d entries instead of %d") %
                ii % params[ii].size() % nSpatialParams);
        }
    }
    // Set parameters
    for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
        this->_spatialFunctionList[ii]->setParameters(params[ii]);
    }
}

/**
 * \brief Compute the kernel parameters at a specified point
 *
 * Warning: this is a low-level function that assumes:
 * * there is a spatial model
 * * kernelParams is the right length
 * It will fail in unpredictable ways if either condition is not met.
 * The only reason it is not protected is because the convolveLinear function needs it.
 */
void lsst::afw::math::Kernel::computeKernelParametersFromSpatialModel(std::vector<double> &kernelParams, double x, double y) const {
    std::vector<double>::iterator kParamsIter = kernelParams.begin();
    std::vector<SpatialFunctionPtr>::const_iterator spFuncIter = _spatialFunctionList.begin();
    for ( ; kParamsIter != kernelParams.end(); ++kParamsIter, ++spFuncIter) {
        *kParamsIter = (*(*spFuncIter))(x,y);
    }
}
