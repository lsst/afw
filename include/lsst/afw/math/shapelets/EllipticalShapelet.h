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
 
#ifndef LSST_AFW_MATH_SHAPELETS_ELLIPTICALSHAPELET_H
#define LSST_AFW_MATH_SHAPELETS_ELLIPTICALSHAPELET_H

/**
 * @file
 *
 * @brief Constants and typedefs for shapelets library.
 *
 * @todo
 *
 * @author Jim Bosch
 */

#include "lsst/ndarray.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/math/shapelets/UnitShapelet.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

class EllipticalShapeletEvaluator;

/**
 *  @brief A shapelet expansion with basis defined on an ellipse.
 */
class EllipticalShapeletFunction {
public:

    /// @brief Return the maximum order (inclusive), either @f$n_x + n_y@f$ or @f$p + q@f$.
    int getOrder() const { return _unit.getOrder(); }

    /// @brief Return the basis type (HERMITE or LAGUERRE).
    BasisTypeEnum getBasisType() const { return _unit.getBasisType(); }

    /// @brief Change the basis type and convert coefficients correspondingly.
    void changeBasisType(BasisTypeEnum basisType) { _unit.changeBasisType(basisType); }

    /// @brief Get the basis ellipse.
    geom::ellipses::Quadrupole const & getEllipse() const { return _ellipse; }

    /// @brief Set the basis ellipse.
    void setEllipse(geom::ellipses::Quadrupole const & ellipse) const { _ellipse = ellipse; }

    /// @brief Return the coefficient vector.
    ndarray::Array<Pixel,1,1> const getCoefficients() { return _unit.getCoefficients(); }

    /// @brief Return the coefficient vector (const).
    ndarray::Array<Pixel,1,1> const getCoefficients() const { return _unit.getCoefficients(); }

    /// @brief Construct a helper object that can efficiently evaluate the function.
    EllipticalShapeletEvaluator evaluate() const;

    /// @brief Construct a function and set all coefficients to zero.
    UnitShapeletFunction(int order, BasisTypeEnum basisType, geom::ellipses::Quadrupole const & ellipse) :
        _unit(order, basisType), _ellipse(ellipse)
    {}

    /// @brief Construct a function with a shallow-copied coefficient vector.
    UnitShapeletFunction(
        int order, BasisTypeEnum basisType,
        ndarray::Array<double,1,1> const & coefficients
    ) : _unit(order, basisType, coefficients), _ellipse(ellipse) {}

private:
    friend class EllipticalShapeletEvaluator;

    UnitShapeletFunction _unit;
    geom::elipses::Quadrupole _ellipse;
};

/**
 *  @brief A shapelet basis defined on the elliptical circle.
 */
class EllipticalShapeletBasis {
public:

    /// @brief Return the maximum order (inclusive), either @f$n_x + n_y@f$ or @f$p + q@f$.
    int getOrder() const { return _unit.getOrder(); }

    /// @brief Return the basis type (HERMITE or LAGUERRE).
    BasisTypeEnum getBasisType() const { return _unit.getBasisType(); }

    /// @brief Set the basis type (HERMITE or LAGUERRE).
    void setBasisType(BasisTypeEnum basisType) { _unit.setBasisType(basisType); }

    /// @brief Get the basis ellipse.
    geom::ellipses::Quadrupole const & getEllipse() const { return _ellipse; }

    /// @brief Set the basis ellipse.
    void setEllipse(geom::ellipses::Quadrupole const & ellipse) const {
        _ellipse = ellipse;
        _transform = _ellipse.getGridTransform();
    }

    /**
     *  @brief Fill the given array with the result of evaluating the basis functions
     *         at the given point.
     */
    void fillEvaluationVector(
        ndarray::Array<Pixel,1,1> const & result,
        double x, double y
    ) const {
        fillEvaluationVector(result, geom::Point2D(x, y));
    }

    /**
     *  @brief Fill the given array with the result of evaluating the basis functions
     *         at the given point.
     */
    void fillEvaluationVector(
        ndarray::Array<Pixel,1,1> const & result,
        geom::Point2D const & point
    ) const {
        _unit.fillEvaluationVector(result, _transform(point));
    }

    /**
     *  @brief Fill the given array the with result of evaluating the basis functions
     *         at the given point.
     */
    void fillEvaluationVector(
        ndarray::Array<Pixel,1,1> const & result,
        geom::Extent2D const & point
    ) const {
        _unit.fillEvaluationVector(result, _transform(point));
    }

    /**
     *  @brief Fill the given array with the result of integrating the basis functions.
     */
    void fillIntegrationVector(ndarray::Array<Pixel,1,1> const & result) const {
        _unit.fillEvaluationVector(result);
        result.deep() *= _ellipse.getArea();
    }

    /// @brief Construct a basis
    EllipticalShapeletBasis(int order, BasisTypeEnum basisType, geom::ellipses::Quadrupole const & ellipse) :
        _unit(order, basisType), _ellipse(ellipse), _transform(_ellipse.getGridTransform())
    {}

private:
    UnitShapeletBasis _unit;
    geom::elipses::Quadrupole _ellipse;
    geom::LinearTransform _transform;
};

/**
 *  @brief Evaluates a EllipticalShapeletFunction.
 *
 *  A EllipticalShapeletEvaluator is invalidated whenever the EllipticalShapeletFunction it
 *  was constructed from is modified.
 */
class EllipticalShapeletEvaluator {
public:

    /// @brief Evaluate at the given point.
    double operator()(double x, double y) const {
        return this->operator()(geom::Point2D(x, y));
    }

    /// @brief Evaluate at the given point.
    double operator()(geom::Point2D const & point) const {
        return _unit(_transform(point)); 
    }

    /// @brief Evaluate at the given point.
    double operator()(geom::Extent2D const & point) const {
        return _unit(_transform(point)); 
    }

    /// @brief Compute the definite integral or integral moments.
    double integrate() const {
        return _unit.integrate() / std::sqrt(_transform.getDeterminant());
    }

    /// @brief Update the evaluator from using the given function.
    void update(EllipticalShapeletFunction const & function) {
        _unit.update(function._unit);
        _transform = function.getEllipse().getGridTransform();
    }

    /// @brief Construct an evaluator for the given function.
    explicit EllipticalShapeletEvaluator(EllipticalShapeletFunction const & function) :
        _unit(function._unit), _transform(function.getEllipse().getGridTransform())
    {}

private:
    UnitShapeletEvaluator _unit;
    geom::LinearTransform _transform;
};

inline EllipticalShapeletEvaluator EllipticalShapeletFunction::evaluate() const {
    return EllipticalShapeletEvaluator(*this);
}

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_ELLIPTICALSHAPELET_H)
