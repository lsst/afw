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
 
#ifndef LSST_AFW_MATH_SHAPELETS_SHAPELETFUNCTION_H
#define LSST_AFW_MATH_SHAPELETS_SHAPELETFUNCTION_H

/**
 * @file
 *
 * @brief A 2-d function defined by an expansion onto a Gauss-Laguerre or Gauss-Hermite basis.
 *
 * @todo
 *
 * @author Jim Bosch
 */

#include "lsst/ndarray.h"
#include "lsst/afw/math/shapelets/constants.h"
#include "lsst/afw/math/shapelets/detail/HermiteEvaluator.h"
#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/afw/geom.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

class ShapeletFunctionEvaluator;

/**
 *  @brief A 2-d function defined by an expansion onto a Gauss-Laguerre or Gauss-Hermite basis.
 */
class ShapeletFunction {
public:

    typedef boost::shared_ptr<ShapeletFunction> Ptr;
    typedef boost::shared_ptr<ShapeletFunction const> ConstPtr;

    /// @brief Return the maximum order (inclusive), either @f$n_x + n_y@f$ or @f$p + q@f$.
    int getOrder() const { return _order; }

    /// @brief Get the ellipse (const).
    EllipseCore const & getEllipse() const { return _ellipse; }

    /// @brief Get the ellipse (non-const).
    EllipseCore & getEllipse() { return _ellipse; }

    /// @brief Set the ellipse.
    void setEllipse(EllipseCore const & ellipse) { _ellipse = ellipse; }
    
    /// @brief Return the radius of the ellipse.
    double getRadius() const { return _ellipse.getRadius(); }
    
    /// @brief Set the radius of the ellipse.
    void setRadius(double radius) { _ellipse.setRadius(radius); }

    /// @brief Return the basis type (HERMITE or LAGUERRE).
    BasisTypeEnum getBasisType() const { return _basisType; }

    /// @brief Change the basis type and convert coefficients in-place correspondingly.
    void changeBasisType(BasisTypeEnum basisType) {
        ConversionMatrix::convertCoefficientVector(_coefficients, _basisType, basisType, _order);
        _basisType = basisType;
    }

    /// @brief Return the coefficient vector (always mutable; sharing destroys const-protection).
    lsst::ndarray::Array<Pixel,1,1> const getCoefficients() const { return _coefficients; }

    /// @brief Convolve the shapelet function in-place.
    void convolve(ShapeletFunction const & other);

    /// @brief Construct a helper object that can efficiently evaluate the function.
    ShapeletFunctionEvaluator evaluate() const;

    /// @brief Construct a function with a unit-circle ellipse and set all coefficients to zero.
    ShapeletFunction(int order, BasisTypeEnum basisType);

    /// @brief Construct a function with a unit-circle ellipse and a shallow-copied coefficient vector.
    ShapeletFunction(
        int order, BasisTypeEnum basisType,
        lsst::ndarray::Array<Pixel,1,1> const & coefficients
    );

    /// @brief Construct a function with a circular ellipse and set all coefficients to zero.
    ShapeletFunction(int order, BasisTypeEnum basisType, double radius);

    /// @brief Construct a function with a circular ellipse and a shallow-copied coefficient vector.
    ShapeletFunction(
        int order, BasisTypeEnum basisType, double radius,
        lsst::ndarray::Array<Pixel,1,1> const & coefficients
    );

    /// @brief Construct a function and set all coefficients to zero.
    ShapeletFunction(int order, BasisTypeEnum basisType, EllipseCore const & ellipse);

    /// @brief Construct a function with a shallow-copied coefficient vector.
    ShapeletFunction(
        int order, BasisTypeEnum basisType, EllipseCore const & ellipse,
        lsst::ndarray::Array<Pixel,1,1> const & coefficients
    );

    /// @brief Copy constructor.
    ShapeletFunction(ShapeletFunction const & other, bool deep = false);

private:
    int _order;
    BasisTypeEnum _basisType;
    EllipseCore _ellipse;
    lsst::ndarray::Array<Pixel,1,1> _coefficients;
};

/**
 *  @brief Evaluates a ShapeletFunction.
 *
 *  A ShapeletFunctionEvaluator is invalidated whenever the ShapeletFunction it
 *  was constructed from is modified.
 */
class ShapeletFunctionEvaluator {
public:

    typedef boost::shared_ptr<ShapeletFunctionEvaluator> Ptr;
    typedef boost::shared_ptr<ShapeletFunctionEvaluator const> ConstPtr;

    /// @brief Evaluate at the given point.
    Pixel operator()(double x, double y) const {
        return this->operator()(geom::Point2D(x, y));
    }

    /// @brief Evaluate at the given point.
    Pixel operator()(geom::Point2D const & point) const {
        return _h.sumEvaluation(_coefficients, _transform(point));
    }

    /// @brief Evaluate at the given point.
    Pixel operator()(geom::Extent2D const & point) const {
        return _h.sumEvaluation(_coefficients, _transform(point));
    }

    /// @brief Compute the definite integral or integral moments.
    Pixel integrate() const {
        return _h.sumIntegration(_coefficients) / std::sqrt(_transform.computeDeterminant());
    }

    /// @brief Update the evaluator from the given function.
    void update(ShapeletFunction const & function);

    /// @brief Construct an evaluator for the given function.
    explicit ShapeletFunctionEvaluator(ShapeletFunction const & function);

private:
    
    void _initialize(ShapeletFunction const & function);
    ndarray::Array<Pixel,1,1> _coefficients;
    geom::LinearTransform _transform;
    detail::HermiteEvaluator _h;
};

inline ShapeletFunctionEvaluator ShapeletFunction::evaluate() const {
    return ShapeletFunctionEvaluator(*this);
}

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_SHAPELETFUNCTION_H)
