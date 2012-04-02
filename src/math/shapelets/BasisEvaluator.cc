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

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/shapelets/BasisEvaluator.h"

namespace lsst { namespace afw { namespace math { namespace shapelets {

namespace {

static inline void validateSize(int expected, int actual) {
    if (expected != actual) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthErrorException,
            (boost::format(
                "Output array for BasisEvaluator has incorrect size (%n, should be %n)."
            ) % actual % expected).str()
        );
    }
}

} // anonymous

void BasisEvaluator::fillEvaluation(
    ndarray::Array<Pixel,1> const & array, double x, double y,
    ndarray::Array<Pixel,1> const & dx,
    ndarray::Array<Pixel,1> const & dy
) const {
    validateSize(computeSize(getOrder()), array.getSize<0>());
    _h.fillEvaluation(array, x, y, dx, dy);
    ConversionMatrix::convertOperationVector(array, HERMITE, _basisType, getOrder());
    if (!dx.isEmpty()) {
        validateSize(computeSize(getOrder()), dx.getSize<0>());
        ConversionMatrix::convertOperationVector(dx, HERMITE, _basisType, getOrder());
    }
    if (!dy.isEmpty()) {
        validateSize(computeSize(getOrder()), dy.getSize<0>());
        ConversionMatrix::convertOperationVector(dy, HERMITE, _basisType, getOrder());
    }
}

void BasisEvaluator::fillIntegration(ndarray::Array<Pixel,1> const & array, int xMoment, int yMoment) const {
    validateSize(computeSize(getOrder()), array.getSize<0>());
    _h.fillIntegration(array, xMoment, yMoment);
    ConversionMatrix::convertOperationVector(array, HERMITE, _basisType, getOrder());
}

}}}} // namespace lsst::afw::math::shapelets
