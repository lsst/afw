// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
/**
 * \file
 *
 * \brief Definition of inline member functions declared in Kernel.h
 *
 * This file is meant to be included by lsst/afw/math/Kernel.h
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <lsst/pex/exceptions.h>

namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

//
// Inline Member Functions
//

/**
 * \brief Return the number of kernel parameters (0 if none)
 */
inline unsigned afwMath::Kernel::getNKernelParameters() const {
    return _nKernelParams;
}

/**
 * \brief Return the number of spatial parameters (0 if not spatially varying)
 */
inline unsigned afwMath::Kernel::getNSpatialParameters() const { 
    if (!isSpatiallyVarying()) {
        return 0;
    } else {
        return _spatialFunctionPtr->getNParameters();
    }
}

/**
 * \brief Return the spatial parameters parameters (an empty vector if not spatially varying)
 */
inline std::vector<std::vector<double> > afwMath::Kernel::getSpatialParameters() const {
    return _spatialParams;
}

/**
 * \brief Return true if the kernel is spatially varying (has a spatial function)
 */
inline bool afwMath::Kernel::isSpatiallyVarying() const {
    return _isSpatiallyVarying;
}

/**
 * \brief Set the kernel parameters of a spatially invariant kernel.
 *
 * Note: if lsst::pex::exceptions::RuntimeError becomes available then 
 * I plan to use that instead of lsst::pex::exceptions::Exception.
 *
 * \throw lsst::pex::exceptions::Runtime if the kernel has a spatial function
 * \throw lsst::pex::exceptions::InvalidParameter if the params vector is the wrong length
 */
inline void afwMath::Kernel::setKernelParameters(std::vector<double> const &params) {
    if (this->isSpatiallyVarying()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::Runtime, "Kernel is spatially varying");
    }
    this->basicSetKernelParameters(params);
}
