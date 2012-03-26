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

#include "ndarray.h"
#include "lsst/afw/math/shapelets/constants.h"
#include "lsst/afw/math/shapelets/HermiteEvaluator.h"
#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"

#include <list>

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

    typedef ShapeletFunctionEvaluator Evaluator;

    /// @brief Return the maximum order (inclusive), either @f$n_x + n_y@f$ or @f$p + q@f$.
    int getOrder() const { return _order; }

    /// @brief Get the ellipse (const).
    lsst::afw::geom::ellipses::Ellipse const & getEllipse() const { return _ellipse; }

    /// @brief Get the ellipse (non-const).
    lsst::afw::geom::ellipses::Ellipse & getEllipse() { return _ellipse; }

    /// @brief Set the ellipse.
    void setEllipse(lsst::afw::geom::ellipses::Ellipse const & ellipse) { _ellipse = ellipse; }
    
    /// @brief Return the basis type (HERMITE or LAGUERRE).
    BasisTypeEnum getBasisType() const { return _basisType; }

    /// @brief Change the basis type and convert coefficients in-place correspondingly.
    void changeBasisType(BasisTypeEnum basisType) {
        ConversionMatrix::convertCoefficientVector(_coefficients, _basisType, basisType, _order);
        _basisType = basisType;
    }

    /// @brief Normalize the integral of the shapelet function to 1.
    void normalize();

    /// @brief Return the coefficient vector.
    ndarray::Array<Pixel,1,1> const getCoefficients() { return _coefficients; }

    /// @brief Return the coefficient vector (const).
    ndarray::Array<Pixel const,1,1> const getCoefficients() const { return _coefficients; }

    /// @brief Convolve the shapelet function in-place.
    void convolve(ShapeletFunction const & other);

    /// @brief Construct a helper object that can efficiently evaluate the function.
    Evaluator evaluate() const;

    /// @brief Shift the shapelet function by shifting the ellipse of each element.
    void shiftInPlace(lsst::afw::geom::Extent2D const & offset) {
        _ellipse.getCenter() += offset;
    }

    /// @brief Transform the shapelet function by transforming the ellipse of each elements.
    void transformInPlace(lsst::afw::geom::AffineTransform const & transform) {
        _ellipse.transform(transform).inPlace();
    }

    /// @brief Construct a function with a unit-circle ellipse and set all coefficients to zero.
    ShapeletFunction(int order, BasisTypeEnum basisType);

    /// @brief Construct a function with a unit-circle ellipse and a deep-copied coefficient vector.
    ShapeletFunction(
        int order, BasisTypeEnum basisType,
        ndarray::Array<lsst::afw::math::shapelets::Pixel,1,1> const & coefficients
    );

    /// @brief Construct a function with a circular ellipse and set all coefficients to zero.
    ShapeletFunction(int order, BasisTypeEnum basisType, double radius,
                     lsst::afw::geom::Point2D const & center);

    /// @brief Construct a function with a circular ellipse and a deep-copied coefficient vector.
    ShapeletFunction(
        int order, BasisTypeEnum basisType, double radius, lsst::afw::geom::Point2D const & center,
        ndarray::Array<lsst::afw::math::shapelets::Pixel,1,1> const & coefficients
    );

    /// @brief Construct a function and set all coefficients to zero.
    ShapeletFunction(int order, BasisTypeEnum basisType,
        lsst::afw::geom::ellipses::Ellipse const & ellipse);

    /// @brief Construct a function with a deep-copied coefficient vector.
    ShapeletFunction(
        int order, BasisTypeEnum basisType,
        lsst::afw::geom::ellipses::Ellipse const & ellipse,
        ndarray::Array<lsst::afw::math::shapelets::Pixel,1,1> const & coefficients
    );

    /// @brief Copy constructor (deep).
    ShapeletFunction(ShapeletFunction const & other);

    /// @brief Assignment (deep).
    ShapeletFunction & operator=(ShapeletFunction const & other);

private:

    friend class std::list<ShapeletFunction>;

    /// @brief Default constructor to appease SWIG (used by std::list).
    ShapeletFunction();

    int _order;
    BasisTypeEnum _basisType;
    lsst::afw::geom::ellipses::Ellipse _ellipse;
    ndarray::Array<Pixel,1,1> _coefficients;
};

/**
 *  @brief Evaluates a ShapeletFunction.
 *
 *  This is distinct from ShapeletFunction to allow the evaluator to construct temporaries
 *  and allocate workspace that will be reused when evaluating at multiple points.
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
    Pixel operator()(lsst::afw::geom::Point2D const & point) const {
        return _h.sumEvaluation(_coefficients, _transform(point));
    }

    /// @brief Evaluate at the given point.
    Pixel operator()(lsst::afw::geom::Extent2D const & point) const {
        return _h.sumEvaluation(_coefficients, _transform(point));
    }

    /// @brief Compute the definite integral or integral moments.
    Pixel integrate() const {
        return _h.sumIntegration(_coefficients) / _transform.getLinear().computeDeterminant();
    }

    /// @brief Return the unweighted dipole and quadrupole moments of the function as an ellipse.
    lsst::afw::geom::ellipses::Ellipse computeMoments() const;

    /// @brief Update the evaluator from the given function.
    void update(ShapeletFunction const & function);

    /// @brief Construct an evaluator for the given function.
    explicit ShapeletFunctionEvaluator(ShapeletFunction const & function);

private:
    
    friend class MultiShapeletFunctionEvaluator;

    void _initialize(ShapeletFunction const & function);

    void _computeRawMoments(double & q0, Eigen::Vector2d & q1, Eigen::Matrix2d & q2) const;

    ndarray::Array<Pixel const,1,1> _coefficients;
    geom::AffineTransform _transform;
    HermiteEvaluator _h;
};

inline ShapeletFunctionEvaluator ShapeletFunction::evaluate() const {
    return ShapeletFunctionEvaluator(*this);
}

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_SHAPELETFUNCTION_H)
