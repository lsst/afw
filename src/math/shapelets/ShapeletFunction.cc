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
#include "lsst/afw/math/shapelets/detail/HermiteConvolution.h"
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
    _order(order), _basisType(basisType), _ellipse(0.0, 0.0, 1.0),
    _coefficients(nd::copy(coefficients))
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
    _order(order), _basisType(basisType), _ellipse(0.0, 0.0, radius),
    _coefficients(nd::copy(coefficients))
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
    _order(order), _basisType(basisType), _ellipse(ellipse),
    _coefficients(nd::copy(coefficients))
{
    validateSize(computeSize(order), _coefficients.getSize<0>());
}

shapelets::ShapeletFunction::ShapeletFunction(ShapeletFunction const & other) :
    _order(other._order), _basisType(other._basisType), _ellipse(other._ellipse),
    _coefficients(nd::copy(other._coefficients))
{}

shapelets::ShapeletFunction & shapelets::ShapeletFunction::operator=(ShapeletFunction const & other) {
    if (&other != this) {
        if (other.getOrder() != this->getOrder()) {
            _order = other.getOrder();
            _coefficients = ndarray::copy(other.getCoefficients());
        } else {
            _coefficients.deep() = other.getCoefficients();
        }
        _basisType = other.getBasisType();
        _ellipse = other.getEllipse();
    }
    return *this;
}

void shapelets::ShapeletFunctionEvaluator::update(ShapeletFunction const & function) {
    validateSize(_h.getOrder(), function.getOrder());
    _transform = function.getEllipse().getGridTransform();
    _initialize(function);
}

shapelets::ShapeletFunctionEvaluator::ShapeletFunctionEvaluator(
    shapelets::ShapeletFunction const & function
) : _transform(function.getEllipse().getGridTransform()), _h(function.getOrder()) {
    _initialize(function);
}

void shapelets::ShapeletFunctionEvaluator::_initialize(ShapeletFunction const & function) {
    switch (function.getBasisType()) {
    case HERMITE:
        _coefficients = function.getCoefficients();
        break;
    case LAGUERRE:
        nd::Array<Pixel,1,1> tmp(nd::copy(function.getCoefficients()));
        ConversionMatrix::convertCoefficientVector(
            tmp, shapelets::LAGUERRE, shapelets::HERMITE, function.getOrder()
        );
        _coefficients = tmp;
        break;
    }
}

void shapelets::ShapeletFunction::convolve(shapelets::ShapeletFunction const & other) {
    detail::HermiteConvolution convolution(other.getOrder(), *this);
    ndarray::EigenView<Pixel const,2,2> matrix(convolution.evaluate(_ellipse));
    if (_basisType == LAGUERRE) {
        ConversionMatrix::convertCoefficientVector(_coefficients, LAGUERRE, HERMITE, getOrder());
    }
    ndarray::viewAsEigen(_coefficients) = matrix * ndarray::viewAsEigen(_coefficients);
    if (_basisType == LAGUERRE) {
        ConversionMatrix::convertCoefficientVector(_coefficients, HERMITE, LAGUERRE, getOrder());
    }
}
