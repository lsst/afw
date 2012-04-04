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
 
#ifndef LSST_AFW_MATH_SHAPELETS_BASISEVALUATOR_H
#define LSST_AFW_MATH_SHAPELETS_BASISEVALUATOR_H

#include "ndarray.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/shapelets/constants.h"
#include "lsst/afw/math/shapelets/HermiteEvaluator.h"
#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/detection/Footprint.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

/**
 *  @brief Evaluates a standard shapelet Basis.
 */
class BasisEvaluator {
public:

    /// @brief Construct an evaluator for a shapelet basis with the given order and type.
    BasisEvaluator(int order, BasisTypeEnum basisType) : _basisType(basisType), _h(order) {}

    /// @brief Return the order of the shapelet expansion.
    int getOrder() const { return _h.getOrder(); }

    /// @brief Return the type of the shapelet expansion.
    BasisTypeEnum getBasisType() const { return _basisType; }

    /**
     *  @brief Fill an array with an evaluation vector that can be used to evaluate a shapelet model
     *         at a point.
     *
     *  @param[out]    array    Output array.  Must be preallocated to the correct size.
     *  @param[in]     x        x coordinate at which to evaluate the basis.
     *  @param[in]     y        y coordinate at which to evaluate the basis.
     *  @param[out]    dx       Optional output array for the derivative w.r.t. the x coordinate.
     *  @param[out]    dy       Optional output array for the derivative w.r.t. the y coordinate.
     */
    void fillEvaluation(
        ndarray::Array<Pixel,1> const & array, double x, double y,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const;

    /**
     *  @brief Fill an array with an evaluation vector that can be used to evaluate a shapelet model
     *         at a point.
     *
     *  @param[out]    array    Output array.  Must be preallocated to the correct size.
     *  @param[in]     point    Coordinates at which to evaluate the basis.
     *  @param[out]    dx       Optional output array for the derivative w.r.t. the x coordinate.
     *  @param[out]    dy       Optional output array for the derivative w.r.t. the y coordinate.
     */
    void fillEvaluation(
        ndarray::Array<Pixel,1> const & array, geom::Point2D const & point,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const {
        fillEvaluation(array, point.getX(), point.getY(), dx, dy);
    }

    /**
     *  @brief Fill an array with an evaluation vector that can be used to evaluate a shapelet model
     *         at a point.
     *
     *  @param[out]    array    Output array.  Must be preallocated to the correct size.
     *  @param[in]     point    Coordinates at which to evaluate the basis.
     *  @param[out]    dx       Optional output array for the derivative w.r.t. the x coordinate.
     *  @param[out]    dy       Optional output array for the derivative w.r.t. the y coordinate.
     */
    void fillEvaluation(
        ndarray::Array<Pixel,1> const & array, geom::Extent2D const & point,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const {
        fillEvaluation(array, point.getX(), point.getY(), dx, dy);
    }

    /**
     *  @brief Fill an array with an integration vector that can be used to integrate a shapelet model.
     *
     *  @param[out]    array    Output array.  Must be preallocated to the correct size.
     *  @param[in]     xMoment  Integrate the expansion multiplied by this power of x.
     *  @param[in]     yMoment  Integrate the expansion multiplied by this power of y.
     */
    void fillIntegration(ndarray::Array<Pixel,1> const & array, int xMoment=0, int yMoment=0) const;

private:
    BasisTypeEnum _basisType;
    HermiteEvaluator _h;
};

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_BASISEVALUATOR_H)
