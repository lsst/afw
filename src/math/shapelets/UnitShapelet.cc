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

#include "lsst/afw/math/shapelets/UnitShapelet.h"
#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/pex/exceptions.h"
#include "lsst/ndarray/eigen.h"
#include <boost/format.hpp>

namespace shapelets = lsst::afw::math::shapelets;
namespace nd = lsst::ndarray;

namespace {

static double const NORMALIZATION = std::pow(M_PI, -0.25);

/**
 *  @brief An iterator-like object to help in traversing "packed" shapelet or Hermite polynomial
 *         matrix or vector dimensions.
 *
 *  A pair of indices (x,y) is mapped to the packed position i = (x+y)(x+y+1)/2 + x.
 *
 *  Typical usage is in a nested loop of the form:
 *  @code
 *      for (PackedIndex i; i.getOrder() <= order; ++i) {
 *          // utilize i
 *      }
 *  @endcode
 */
class PackedIndex {
public:
    
    static int const computeOffset(int order) { return order*(order+1)/2; }
    static int const computeIndex(int x, int y) { return computeOffset(x+y) + x; }

    PackedIndex & operator++() {
        ++_i;
        if (--_y < 0) {
            _x = 0;
            _y = ++_n;
        } else {
            ++_x;
        }
        return *this;
    }

    int const getOrder() const { return _n; }
    int const getX() const { return _x; }
    int const getY() const { return _y; }

    int const getIndex() const { return _i; }

    PackedIndex() : _n(0), _i(0), _x(0), _y(0) {}
    PackedIndex(int const x, int const y) : _n(x+y), _i(computeOffset(_n) + x), _x(x), _y(y) {}

private:
    int _n;
    int _i;
    int _x;
    int _y;
};

/**
 *  @brief An iterator-like construct for evaluating either normalized Hermite polynomials or
 *         Gauss-Hermite functions at a point.
 */
class HermiteRecurrenceRelation {
public:

    static int const computeSize(int order) { return order + 1; }

    /// @brief Return the value of the shapelet basis function at the current order.
    double operator()() const { return _current; }

    /// @brief Return the derivative of the shapelet basis function at the current order.
    double derivative() const { return std::sqrt(2.0*_order) * _previous - _x * _current; }

    /// @brief Increase the order of the recurrence relation by one.
    HermiteRecurrenceRelation & operator++() {
        double copy = _current;
        ++_order;
        _current = std::sqrt(2.0/_order)*_x*_current - std::sqrt((_order - 1.0)/_order)*_previous;
        _previous = copy;
        return *this;
    }

    /// @brief Return the current order of the recursion relation.
    int const getOrder() const { return _order; }

    /// @brief Initialize a recurrence relation at the given point.
    HermiteRecurrenceRelation(double x, double amplitude) :
        _order(0), _x(x), _current(amplitude), _previous(0.0) {}

private:
    int _order;
    double _x;
    double _current;
    double _previous;
};

void fillEvaluationVector1d(nd::Array<shapelets::Pixel,1,1> const & result, double x) {
    HermiteRecurrenceRelation r(x, NORMALIZATION * std::exp(-0.5*x*x));
    nd::Array<shapelets::Pixel,1,1>::Iterator const end = result.end();
    for (nd::Array<shapelets::Pixel,1,1>::Iterator i = result.begin(); i != end; ++i, ++r) {
        *i = r();
    }
}

void fillIntegrationVector1d(nd::Array<shapelets::Pixel,1,1> const & result, int moment) {
    int const order = result.getSize<0>() - 1;
    result.deep() = 0.0;
    result[0] = std::pow(4.0*M_PI, 0.25);
    for (int n = 2; n <= order; n += 2) {
        result[n] = std::sqrt((n - 1.0) / n) * result[n-2];
    }
    if (moment > 0) {
        // since result is only nonzero for (m+n) even, we store both n,m and n-1,m-1 in the same vector
        for (int n = 1; n <= order; n += 2) {
            result[n] = result[n-1] * std::sqrt(2.0*n);
        }
        for (int m = 2; m <= moment; ++m) {
            if (m % 2 == 0) result[0] *= (m-1);
            for (int n = 2 - (m % 2); n <= order; n += 2) {
                result[n] = (m-1) * result[n] + std::sqrt(2.0*n) * result[n-1];
            }
        }
        // zero the elements corresponding to n-1,m-1
        for (int n = !(moment % 2); n <= order; n += 2) result[n] = 0.0;
    }
}

void weaveFill(
    nd::Array<shapelets::Pixel,1> const & result,
    nd::Array<shapelets::Pixel,1,1> const & x,
    nd::Array<shapelets::Pixel,1,1> const & y
) {
    int const order = x.getSize<0>() - 1;
    for (PackedIndex i; i.getOrder() <= order; ++i) {
        result[i.getIndex()] = x[i.getX()] * y[i.getY()];
    }
}

double weaveInnerProduct(
    nd::Array<shapelets::Pixel,1> const & coefficients,
    nd::Array<shapelets::Pixel,1,1> const & x,
    nd::Array<shapelets::Pixel,1,1> const & y
) {
    double r = 0.0;
    int const order = x.getSize<0>() - 1;
    for (PackedIndex i; i.getOrder() <= order; ++i) {
        r += coefficients[i.getIndex()] * x[i.getX()] * y[i.getY()];
    }
    return r;
}

} // anonymous    

shapelets::UnitShapeletFunction::UnitShapeletFunction(int order, BasisTypeEnum basisType) :
    _order(order), _basisType(basisType), _coefficients(nd::allocate(computeSize(_order)))
{
    _coefficients.deep() = 0.0;
}

shapelets::UnitShapeletFunction::UnitShapeletFunction(
    int order, BasisTypeEnum basisType,
    nd::Array<shapelets::Pixel,1,1> const & coefficients
) :
    _order(order), _basisType(basisType), _coefficients(coefficients)
{
    if (computeSize(order) != _coefficients.getSize<0>()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            (boost::format(
                "Coefficient vector for UnitShapeletFunction has incorrect size (%n, should be %n)."
            ) % _coefficients.getSize<0>() % computeSize(order)).str()
        );
    }
}

void shapelets::UnitShapeletBasis::fillEvaluationVector(
    nd::Array<shapelets::Pixel,1> const & result,
    double x, double y
) const {
    fillEvaluationVector1d(_workspaceX, x);
    fillEvaluationVector1d(_workspaceY, y);
    weaveFill(result, _workspaceX, _workspaceY);
    if (_basisType == shapelets::LAGUERRE) {
        ConversionMatrix::convertOperationVector(result, shapelets::HERMITE, shapelets::LAGUERRE, _order);
    }
}

void shapelets::UnitShapeletBasis::fillIntegrationVector(
    nd::Array<Pixel,1> const & result,
    int momentX, int momentY
) const {
    fillIntegrationVector1d(_workspaceX, momentY);
    fillIntegrationVector1d(_workspaceY, momentY);
    weaveFill(result, _workspaceX, _workspaceY);
    if (_basisType == shapelets::LAGUERRE) {
        ConversionMatrix::convertOperationVector(result, shapelets::HERMITE, shapelets::LAGUERRE, _order);
    }
}

double shapelets::UnitShapeletEvaluator::operator()(double x, double y) const {
    fillEvaluationVector1d(_workspaceX, x);
    fillEvaluationVector1d(_workspaceY, y);
    return weaveInnerProduct(_coefficients, _workspaceX, _workspaceY);
}

double shapelets::UnitShapeletEvaluator::integrate(int momentX, int momentY) const {
    fillIntegrationVector1d(_workspaceX, momentY);
    fillIntegrationVector1d(_workspaceY, momentY);
    return weaveInnerProduct(_coefficients, _workspaceX, _workspaceY);
}

void shapelets::UnitShapeletEvaluator::update(shapelets::UnitShapeletFunction const & function) {
    _coefficients = function.getCoefficients();
    if (function.getBasisType() == shapelets::LAGUERRE) {
        _coefficients = nd::copy(_coefficients);
        ConversionMatrix::convertCoefficientVector(
            _coefficients, shapelets::LAGUERRE, shapelets::HERMITE, function.getOrder()
        );
    }
    if (function.getOrder() + 1 != _workspaceX.getSize<0>()) {
        _workspaceX = nd::allocate(function.getOrder() + 1);
        _workspaceY = nd::allocate(function.getOrder() + 1);
    }
}

shapelets::UnitShapeletEvaluator::UnitShapeletEvaluator(shapelets::UnitShapeletFunction const & function) :
    _coefficients(function.getCoefficients()),
    _workspaceX(nd::allocate(function.getOrder() + 1)),
    _workspaceY(nd::allocate(function.getOrder() + 1))
{
    if (function.getBasisType() == shapelets::LAGUERRE) {
        _coefficients = nd::copy(_coefficients);
        ConversionMatrix::convertCoefficientVector(
            _coefficients, shapelets::LAGUERRE, shapelets::HERMITE, function.getOrder()
        );
    }
}

