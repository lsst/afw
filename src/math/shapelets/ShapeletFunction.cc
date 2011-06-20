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
#include "lsst/afw/math/shapelets/HermiteConvolution.h"
#include "lsst/pex/exceptions.h"
#include "lsst/ndarray/eigen.h"
#include <boost/format.hpp>

namespace shapelets = lsst::afw::math::shapelets;
namespace geom = lsst::afw::geom;
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

shapelets::ShapeletFunction::ShapeletFunction() : 
    _order(0), _basisType(HERMITE),
    _ellipse(EllipseCore(0.0, 0.0, 1.0), geom::Point2D()), 
    _coefficients(ndarray::allocate(1))
{
    _coefficients[0] = 0.0;
}

shapelets::ShapeletFunction::ShapeletFunction(int order, BasisTypeEnum basisType) :
    _order(order), _basisType(basisType), _ellipse(EllipseCore(0.0, 0.0, 1.0)),
    _coefficients(nd::allocate(computeSize(_order)))
{
    _coefficients.deep() = 0.0;
}

shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType,
    lsst::ndarray::Array<lsst::afw::math::shapelets::Pixel,1,1> const & coefficients
) :
    _order(order), _basisType(basisType), _ellipse(EllipseCore(0.0, 0.0, 1.0)),
    _coefficients(nd::copy(coefficients))
{
    validateSize(computeSize(order), _coefficients.getSize<0>());
}
 
shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType, double radius,
    geom::Point2D const & center
) :
    _order(order), _basisType(basisType), _ellipse(EllipseCore(0.0, 0.0, radius), center),
    _coefficients(nd::allocate(computeSize(_order)))
{
    _coefficients.deep() = 0.0;
}

shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType, double radius, geom::Point2D const & center,
    lsst::ndarray::Array<lsst::afw::math::shapelets::Pixel,1,1> const & coefficients
) :
    _order(order), _basisType(basisType), _ellipse(EllipseCore(0.0, 0.0, radius), center),
    _coefficients(nd::copy(coefficients))
{
    validateSize(computeSize(order), _coefficients.getSize<0>());
}
 
shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType, geom::ellipses::Ellipse const & ellipse
) :
    _order(order), _basisType(basisType), _ellipse(EllipseCore(ellipse.getCore()), ellipse.getCenter()),
    _coefficients(nd::allocate(computeSize(_order)))
{
    _coefficients.deep() = 0.0;
}

shapelets::ShapeletFunction::ShapeletFunction(
    int order, BasisTypeEnum basisType, geom::ellipses::Ellipse const & ellipse,
    lsst::ndarray::Array<lsst::afw::math::shapelets::Pixel,1,1> const & coefficients
) :
    _order(order), _basisType(basisType), _ellipse(EllipseCore(ellipse.getCore()), ellipse.getCenter()),
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

void shapelets::ShapeletFunction::normalize() {
    _coefficients.deep() /= evaluate().integrate();
}

void shapelets::ShapeletFunctionEvaluator::update(ShapeletFunction const & function) {
    validateSize(_h.getOrder(), function.getOrder());
    _transform = function.getEllipse().getGridTransform();
    _initialize(function);
}

shapelets::ShapeletFunctionEvaluator::ShapeletFunctionEvaluator(
    lsst::afw::math::shapelets::ShapeletFunction const & function
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

shapelets::ShapeletFunction shapelets::ShapeletFunction::convolve(
    lsst::afw::math::shapelets::ShapeletFunction const & other
) const {
    HermiteConvolution convolution(other.getOrder(), *this);
    ShapeletFunction result(convolution.getRowOrder(), getBasisType(), _ellipse);
    ndarray::EigenView<Pixel const,2,2> matrix(convolution.evaluate(result.getEllipse()));
    if (_basisType == LAGUERRE) {
        ConversionMatrix::convertCoefficientVector(_coefficients, LAGUERRE, HERMITE, getOrder());
    }
    ndarray::viewAsEigen(result.getCoefficients()) = matrix * ndarray::viewAsEigen(_coefficients);
    if (_basisType == LAGUERRE) {
        ConversionMatrix::convertCoefficientVector(_coefficients, HERMITE, LAGUERRE, getOrder());
        ConversionMatrix::convertCoefficientVector(
            result.getCoefficients(), HERMITE, LAGUERRE, result.getOrder()
        );
    }
    return result;
}

void shapelets::ShapeletFunctionEvaluator::_computeRawMoments(
    double & q0, Eigen::Vector2d & q1, Eigen::Matrix2d & q2
) const {
    double determinant = _transform.getLinear().computeDeterminant();
    Eigen::Matrix2d a = _transform.getLinear().invert().getMatrix();
    Eigen::Vector2d b = _transform.getTranslation().asEigen();

    double m0 = _h.sumIntegration(_coefficients, 0, 0);
    q0 += m0 / determinant;

    Eigen::Vector2d m1(
        _h.sumIntegration(_coefficients, 1, 0),
        _h.sumIntegration(_coefficients, 0, 1)
    );
    q1 += a * (m1 - b * m0) / determinant;

    Eigen::Matrix2d m2;
    m2(0, 0) = _h.sumIntegration(_coefficients, 2, 0);
    m2(1, 1) = _h.sumIntegration(_coefficients, 0, 2);
    m2(0, 1) = m2(1, 0) = _h.sumIntegration(_coefficients, 1, 1);
    q2 += a * (m2 + b * b.transpose() * m0 - m1 * b.transpose() - b * m1.transpose()) * a.transpose() 
        / determinant;
}

geom::Ellipse shapelets::ShapeletFunctionEvaluator::computeMoments() const {
    double q0 = 0.0;
    Eigen::Vector2d q1 = Eigen::Vector2d::Zero();
    Eigen::Matrix2d q2 = Eigen::Matrix2d::Zero();
    _computeRawMoments(q0, q1, q2);
    q1 /= q0;
    q2 /= q0;
    q2 -= q1 * q1.transpose();
    return geom::Ellipse(
        geom::ellipses::Quadrupole(geom::ellipses::Quadrupole::Matrix(q2), false),
        geom::Point2D(q1)
    );
}
