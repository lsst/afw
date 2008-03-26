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
#include <lsst/mwi/exceptions.h>

//
// Inline Member Functions
//

/**
 * \brief Return the number of columns
 */
inline unsigned lsst::fw::Kernel::getCols() const {
    return _cols;
}

/**
 * \brief Return the number of rows
 */
inline unsigned lsst::fw::Kernel::getRows() const {
    return _rows;
}

/**
 * \brief Return the center column
 */
inline unsigned lsst::fw::Kernel::getCtrCol() const {
    return _ctrCol;
}

/**
 * \brief Return the center row
 */
inline unsigned lsst::fw::Kernel::getCtrRow() const {
    return _ctrRow;
}

/**
 * \brief Return the number of kernel parameters (0 if none)
 */
inline unsigned lsst::fw::Kernel::getNKernelParameters() const {
    return _nKernelParams;
}

/**
 * \brief Return the number of spatial parameters (0 if not spatially varying)
 */
inline unsigned lsst::fw::Kernel::getNSpatialParameters() const { 
    if (!isSpatiallyVarying()) {
        return 0;
    } else {
        return _spatialFunctionPtr->getNParameters();
    }
}

/**
 * \brief Return the spatial parameters parameters (an empty vector if not spatially varying)
 */
inline std::vector<std::vector<double> > lsst::fw::Kernel::getSpatialParameters() const {
    return _spatialParams;
}

/**
 * \brief Return true if the kernel is spatially varying (has a spatial function)
 */
inline bool lsst::fw::Kernel::isSpatiallyVarying() const {
    return _isSpatiallyVarying;
}

/**
 * \brief Set the kernel parameters of a spatially invariant kernel.
 *
 * Note: if lsst::mwi::exceptions::RuntimeError becomes available then 
 * I plan to use that instead of lsst::mwi::exceptions::Exception.
 *
 * \throw lsst::mwi::exceptions::Runtime if the kernel has a spatial function
 * \throw lsst::mwi::exceptions::InvalidParameter if the params vector is the wrong length
 */
inline void lsst::fw::Kernel::setKernelParameters(std::vector<double> const &params) {
    if (this->isSpatiallyVarying()) {
        throw lsst::mwi::exceptions::Runtime("Kernel is spatially varying");
    }
    this->basicSetKernelParameters(params);
}
