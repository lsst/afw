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

#include "lsst/afw/math/shapelets/ShapeletFunction.h"
#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/pex/exceptions.h"
#include "lsst/ndarray/eigen.h"
#include <boost/format.hpp>

namespace shapelets = lsst::afw::math::shapelets;
namespace nd = lsst::ndarray;

static inline void validateSize(int expected, int actual) {
    if (expected != actual) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            (boost::format(
                "Coefficient vector for ShapeletFunction has incorrect size (%n, should be %n)."
            ) % actual % expected).str()
        );
    }
}

shapelets::ShapeletFunction::ShapeletFunction(int order, BasisTypeEnum basisType) :
    _order(order), _basisType(basisType), _ellipse(0.0, 0.0, 1.0),
    _coefficients(nd::allocate(computeSize(_order)))
{
    _coefficients.deep() = 0.0;
}

shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType,
    nd::Array<shapelets::Pixel,1,1> const & coefficients
) :
    _order(order), _basisType(basisType), _ellipse(0.0, 0.0, 1.0), _coefficients(coefficients)
{
    validateSize(computeSize(order), _coefficients.getSize<0>());
}
 
shapelets::ShapeletFunction::ShapeletFunction(int order, BasisTypeEnum basisType, double radius) :
    _order(order), _basisType(basisType), _ellipse(0.0, 0.0, radius),
    _coefficients(nd::allocate(computeSize(_order)))
{
    _coefficients.deep() = 0.0;
}

shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType, double radius,
    nd::Array<shapelets::Pixel,1,1> const & coefficients
) :
    _order(order), _basisType(basisType), _ellipse(0.0, 0.0, radius), _coefficients(coefficients)
{
    validateSize(computeSize(order), _coefficients.getSize<0>());
}
 
shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType, EllipseCore const & ellipse
) :
    _order(order), _basisType(basisType), _ellipse(ellipse),
    _coefficients(nd::allocate(computeSize(_order)))
{
    _coefficients.deep() = 0.0;
}

shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType, EllipseCore const & ellipse,
    nd::Array<shapelets::Pixel,1,1> const & coefficients
) :
    _order(order), _basisType(basisType), _ellipse(ellipse), _coefficients(coefficients)
{
    validateSize(computeSize(order), _coefficients.getSize<0>());
}

void shapelets::ShapeletFunctionEvaluator::update(shapelets::ShapeletFunction const & function) {
    validateSize(_h.getOrder(), function.getOrder());
    _transform = function.getEllipse().getGridTransform();
    _initialize(function);
}

shapelets::ShapeletFunctionEvaluator::ShapeletFunctionEvaluator(
    shapelets::ShapeletFunction const & function
) : _transform(function.getEllipse().getGridTransform()), _h(function.getOrder()) {
    _initialize(function);
}

void shapelets::ShapeletFunctionEvaluator::_initialize(shapelets::ShapeletFunction const & function) {
    switch (function.getBasisType()) {
    case HERMITE:
        _h.target = function.getCoefficients();
        break;
    case LAGUERRE:
        _h.target = nd::copy(function.getCoefficients());
        ConversionMatrix::convertCoefficientVector(
            _h.target, shapelets::LAGUERRE, shapelets::HERMITE, function.getOrder()
        );
        break;
    }
}

