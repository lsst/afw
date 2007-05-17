// -*- LSST-C++ -*-
#include <stdexcept>

#include <boost/format.hpp>

// #include <lsst/fw/Kernel.h>

namespace lsst {
namespace fw {

    template<typename PixelT>
    Kernel<PixelT>::Kernel(
    ) :
        LsstBase(typeid(this)),
       _cols(0),
       _rows(0),
       _ctrCol(0),
       _ctrRow(0),
       _nKernelParams(0),
       _isSpatiallyVarying(false),
       _spatialFunctionPtr()
    { }
    
    template<typename PixelT>
    Kernel<PixelT>::Kernel(
        unsigned cols,
        unsigned rows,
        unsigned nKernelParams
    ) :
        LsstBase(typeid(this)),
       _cols(cols),
       _rows(rows),
       _ctrCol(cols/2),
       _ctrRow(cols/2),
       _nKernelParams(nKernelParams),
       _isSpatiallyVarying(false),
       _spatialFunctionPtr()
    { }
    
    template<typename PixelT>
    Kernel<PixelT>::Kernel(
        unsigned cols,
        unsigned rows,
        const unsigned nKernelParams,
        Function2PtrType spatialFunction
    ) :
        LsstBase(typeid(this)),
       _cols(cols),
       _rows(rows),
       _ctrCol(cols/2),
       _ctrRow(cols/2),
       _isSpatiallyVarying(true),
       _spatialFunctionPtr(spatialFunction)
    { }

    /*
     * Return the number of columns
     */
    template<typename PixelT>
    inline const unsigned
    Kernel<PixelT>::getCols() const {
        return _cols;
    }

    /*
     * Return the number of rows
     */
    template<typename PixelT>
    inline const unsigned
    Kernel<PixelT>::getRows() const {
        return _rows;
    }

    /*
     * Return the center column
     */
    template<typename PixelT>
    inline const unsigned
    Kernel<PixelT>::getCtrCol() const {
        return _ctrCol;
    }

    /*
     * Return the center row
     */
    template<typename PixelT>
    inline const unsigned
    Kernel<PixelT>::getCtrRow() const {
        return _ctrRow;
    }
        
    /*
     * Return the kernel parameters at a specified position
     *
     * If the kernel is not spatially varying then the position is ignored
     * If there are no kernel parameters then an empty vector is returned
     */
    template<typename PixelT>
    inline const std::vector<PixelT>
    Kernel<PixelT>::getKernelParameters(
        PixelT x,   ///< x at which to evaluate the spatial model
        PixelT y    ///< y at which to evaluate the spatial model
    ) const {
       if (isSpatiallyVarying()) {
           return getKernelParametersFromSpatialModel(x, y);
       } else {
           return getCurrentKernelParameters();
       }
    }

    /*
     * Return the number of kernel parameters (0 if none)
     */
    template<typename PixelT>
    const unsigned
    Kernel<PixelT>::getNKernelParameters() const {
        return _nKernelParams;
    }

    /*
     * Return the number of spatial parameters (0 if not spatially varying)
     */
    template<typename PixelT>
    inline const unsigned
    Kernel<PixelT>::getNSpatialParameters() const { 
        if (!isSpatiallyVarying()) {
            return 0;
        } else {
            return _spatialFunctionPtr->getNParameters();
        }
    }
    
    /*
     * Return the spatial parameters parameters (an empty vector if not spatially varying)
     */
    template<typename PixelT>
    inline std::vector<std::vector<PixelT> >
    Kernel<PixelT>::getSpatialParameters() const {
        return _spatialParams;
    }

    /*
     * Return true if the kernel is spatially varying (has a spatial function)
     */
    template<typename PixelT>
    inline const bool
    Kernel<PixelT>::isSpatiallyVarying() const {
        return _isSpatiallyVarying;
    }

    /*
     * Set the parameters of the spatial function for all kernel parameters
     *
     * The input is vector[kernel parameter][spatial parameter]
     */
    template<typename PixelT>
    void
    Kernel<PixelT>::setSpatialParameters(const std::vector<std::vector<PixelT> > params) {
        unsigned nKernelParams = this->getNKernelParameters();
        if (params.size() != nKernelParams) {
            throw std::invalid_argument(str(
                boost::format("params has %d entries instead of %d") % params.size() % nKernelParams
            ));
        }
        unsigned nSpatialParams = this->getNSpatialParameters();
        for (unsigned ii = 0; ii < nKernelParams; ++ii) {
            if (params[ii].size() != nSpatialParams) {
                throw std::invalid_argument(str(
                    boost::format("params[%d] has %d entries instead of %d") %
                    ii % params[ii].size() % nSpatialParams
                ));
            }
        }
        _spatialParams = params;
    }

    /*
     * Return the kernel parameters at a specified point
     *
     * Assumes there is a spatial model
     */
    template<typename PixelT>
    const std::vector<PixelT>
    Kernel<PixelT>::getKernelParametersFromSpatialModel(PixelT x, PixelT y) const {
        std::vector<PixelT> kernelParams(getNKernelParameters());
        typename std::vector<PixelT>::iterator kIter;
        typename std::vector<std::vector<PixelT> >::const_iterator sIter;
        for ( kIter = kernelParams.begin(), sIter = _spatialParams.begin();
            kIter != kernelParams.end(); ++kIter, ++sIter) {
            _spatialFunctionPtr->setParameters(*sIter);
            *kIter = (*_spatialFunctionPtr)(x,y);
        }
        return kernelParams;
    }
        
    /*
     * Set the kernel parameters at a specified point
     *
     * Assumes there is a spatial model
     */
    template<typename PixelT>
    void
    Kernel<PixelT>::setKernelParametersFromSpatialModel(PixelT x, PixelT y) {
        setKernelParameters(getKernelParametersFromSpatialModel(x,y));
    }

}   // namespace fw
}   // namespace lsst
