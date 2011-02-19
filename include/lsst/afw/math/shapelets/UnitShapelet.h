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
 
#ifndef LSST_AFW_MATH_SHAPELETS_UNITSHAPELET_H
#define LSST_AFW_MATH_SHAPELETS_UNITSHAPELET_H

/**
 * @file
 *
 * @brief Shapelets with basis defined on the unit circle.
 *
 * @todo
 *
 * @author Jim Bosch
 */

#include "lsst/ndarray.h"
#include "lsst/afw/math/shapelets/constants.h"
#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/afw/geom.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

class UnitShapeletEvaluator;

/**
 *  @brief A shapelet expansion with basis defined on the unit circle.
 */
class UnitShapeletFunction {
public:

    /// @brief Return the maximum order (inclusive), either @f$n_x + n_y@f$ or @f$p + q@f$.
    int getOrder() const { return _order; }

    /// @brief Return the basis type (HERMITE or LAGUERRE).
    BasisTypeEnum getBasisType() const { return _basisType; }
    
    /// @brief Change the basis type and convert coefficients correspondingly.
    void changeBasisType(BasisTypeEnum basisType) {
        ConversionMatrix::convertCoefficientVector(_coefficients, _basisType, basisType, _order);
        _basisType = basisType;
    }

    /// @brief Return the coefficient vector.
    lsst::ndarray::Array<Pixel,1,1> const getCoefficients() { return _coefficients; }

    /// @brief Return the coefficient vector (const).
    lsst::ndarray::Array<Pixel,1,1> const getCoefficients() const { return _coefficients; }

    /// @brief Construct a helper object that can efficiently evaluate the function.
    UnitShapeletEvaluator evaluate() const;

    /// @brief Construct a function and set all coefficients to zero.
    UnitShapeletFunction(int order, BasisTypeEnum basisType);

    /// @brief Construct a function with a shallow-copied coefficient vector.
    UnitShapeletFunction(
        int order, BasisTypeEnum basisType,
        lsst::ndarray::Array<Pixel,1,1> const & coefficients
    );

private:
    int _order;
    BasisTypeEnum _basisType;
    lsst::ndarray::Array<Pixel,1,1> _coefficients;
};

/**
 *  @brief A shapelet basis defined on the unit circle.
 */
class UnitShapeletBasis {
public:

    /// @brief Return the maximum order (inclusive), either @f$n_x + n_y@f$ or @f$p + q@f$.
    int getOrder() const { return _order; }

    /// @brief Return the basis type (HERMITE or LAGUERRE).
    BasisTypeEnum getBasisType() const { return _basisType; }

    /// @brief Set the basis type (HERMITE or LAGUERRE).
    void setBasisType(BasisTypeEnum basisType) { _basisType = basisType; }

    /**
     *  @brief Fill the given array with the result of evaluating the basis functions
     *         at the given point.
     */
    void fillEvaluationVector(
        lsst::ndarray::Array<Pixel,1> const & result,
        double x, double y
    ) const;

    /**
     *  @brief Fill the given array with the result of evaluating the basis functions
     *         at the given point.
     */
    void fillEvaluationVector(
        lsst::ndarray::Array<Pixel,1> const & result,
        geom::Point2D const & point
    ) const {
        fillEvaluationVector(result, point.getX(), point.getY());
    }

    /**
     *  @brief Fill the given array with the result of evaluating the basis functions
     *         at the given point.
     */
    void fillEvaluationVector(
        lsst::ndarray::Array<Pixel,1> const & result,
        geom::Extent2D const & point
    ) const {
        fillEvaluationVector(result, point.getX(), point.getY());
    }

    /**
     *  @brief Fill the given array with the result of integrating the basis functions or
     *         evaluating their unweighted integral moments.
     */
    void fillIntegrationVector(
        lsst::ndarray::Array<Pixel,1> const & result,
        int momentX=0, int momentY=0
    ) const;

    /// @brief Construct a basis
    UnitShapeletBasis(int order, BasisTypeEnum basisType) :
        _order(order), _basisType(basisType),
        _workspaceX(lsst::ndarray::allocate(_order + 1)),
        _workspaceY(lsst::ndarray::allocate(_order + 1))
    {}

private:
    int _order;
    BasisTypeEnum _basisType;
    mutable lsst::ndarray::Array<Pixel,1,1> _workspaceX;
    mutable lsst::ndarray::Array<Pixel,1,1> _workspaceY;
};

/**
 *  @brief Evaluates a UnitShapeletFunction.
 *
 *  A UnitShapeletEvaluator is invalidated whenever the UnitShapeletFunction it
 *  was constructed from is modified.
 */
class UnitShapeletEvaluator {
public:

    /// @brief Evaluate at the given point.
    Pixel operator()(double x, double y) const;

    /// @brief Evaluate at the given point.
    Pixel operator()(geom::Point2D const & point) const {
        return this->operator()(point.getX(), point.getY()); 
    }

    /// @brief Evaluate at the given point.
    Pixel operator()(geom::Extent2D const & point) const {
        return this->operator()(point.getX(), point.getY()); 
    }

    /// @brief Compute the definite integral or integral moments.
    Pixel integrate(int momentX=0, int momentY=0) const;

    /// @brief Update the evaluator from using the given function.
    void update(UnitShapeletFunction const & function);

    /// @brief Construct an evaluator for the given function.
    explicit UnitShapeletEvaluator(UnitShapeletFunction const & function);

private:
    lsst::ndarray::Array<Pixel,1,1> _coefficients;
    mutable lsst::ndarray::Array<Pixel,1,1> _workspaceX;
    mutable lsst::ndarray::Array<Pixel,1,1> _workspaceY;
};

inline UnitShapeletEvaluator UnitShapeletFunction::evaluate() const {
    return UnitShapeletEvaluator(*this);
}

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_UNITSHAPELET_H)
