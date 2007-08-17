// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definition of inline member functions declared in Kernel.h
 *
 * This file is meant to be included by lsst/fw/Kernel.h
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <lsst/mwi/exceptions/Exception.h>

//
// Inline Member Functions
//

/**
 * \brief Return the number of columns
 */
template<typename PixelT>
inline unsigned lsst::fw::Kernel<PixelT>::getCols() const {
    return _cols;
}

/**
 * \brief Return the number of rows
 */
template<typename PixelT>
inline unsigned lsst::fw::Kernel<PixelT>::getRows() const {
    return _rows;
}

/**
 * \brief Return the center column
 */
template<typename PixelT>
inline unsigned lsst::fw::Kernel<PixelT>::getCtrCol() const {
    return _ctrCol;
}

/**
 * \brief Return the center row
 */
template<typename PixelT>
inline unsigned lsst::fw::Kernel<PixelT>::getCtrRow() const {
    return _ctrRow;
}

/**
 * \brief Return the number of kernel parameters (0 if none)
 */
template<typename PixelT>
inline unsigned lsst::fw::Kernel<PixelT>::getNKernelParameters() const {
    return _nKernelParams;
}

/**
 * \brief Return the number of spatial parameters (0 if not spatially varying)
 */
template<typename PixelT>
inline unsigned lsst::fw::Kernel<PixelT>::getNSpatialParameters() const { 
    if (!isSpatiallyVarying()) {
        return 0;
    } else {
        return _spatialFunctionPtr->getNParameters();
    }
}

/**
 * \brief Return the spatial parameters parameters (an empty vector if not spatially varying)
 */
template<typename PixelT>
inline std::vector<std::vector<double> > lsst::fw::Kernel<PixelT>::getSpatialParameters() const {
    return _spatialParams;
}

/**
 * \brief Return true if the kernel is spatially varying (has a spatial function)
 */
template<typename PixelT>
inline bool lsst::fw::Kernel<PixelT>::isSpatiallyVarying() const {
    return _isSpatiallyVarying;
}

/**
 * \brief Set the kernel parameters of a spatially invariant kernel.
 *
 * Note: if lsst::mwi::exceptions::RuntimeError becomes available then 
 * I plan to use that instead of lsst::mwi::exceptions::Exception.
 *
 * \throw lsst::mwi::exceptions::Exception if the kernel has a spatial function
 * \throw lsst::mwi::exceptions::InvalidParameter if the params vector is the wrong length
 */
template<typename PixelT>
inline void lsst::fw::Kernel<PixelT>::setKernelParameters(std::vector<double> const &params) {
    if (this->isSpatiallyVarying()) {
        throw lsst::mwi::exceptions::Exception("Kernel is spatially varying");
    }
    this->basicSetKernelParameters(params);
}
