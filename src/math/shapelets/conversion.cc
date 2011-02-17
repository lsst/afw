// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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

#include "lsst/afw/math/shapelets/constants.h"
#include "lsst/pex/exceptions.h"
#include "lsst/ndarray/eigen.h"
#include <boost/format.hpp>

namespace shapelets = lsst::afw::math::shapelets;
namespace nd = lsst::ndarray;

void shapelets::convertCoefficientVector(
    nd::Array<shapelets::Pixel,1> const & array,
    shapelets::BasisTypeEnum input, shapelets::BasisTypeEnum output, int order
) {
    // TODO (placeholder to allow linker to function)
}

void shapelets::convertOperationVector(
    nd::Array<shapelets::Pixel,1> const & array,
    shapelets::BasisTypeEnum input, shapelets::BasisTypeEnum output, int order
) {
    // TODO (placeholder to allow linker to function)
}
