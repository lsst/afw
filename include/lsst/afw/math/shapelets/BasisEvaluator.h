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
#include "lsst/afw/math/shapelets/HermiteEvaluator.h"
#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/afw/geom.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

/**
 *  @brief Evaluates a Basis.
 *
 *  A BasisEvaluator is invalidated whenever the Basis it
 *  was constructed from is modified.
 */
class BasisEvaluator {
public:

    typedef boost::shared_ptr<BasisEvaluator> Ptr;
    typedef boost::shared_ptr<BasisEvaluator const> ConstPtr;

    /// @brief Construct an evaluator for the given function.
    BasisEvaluator(int order, BasisTypeEnum basisType) : _basisType(basisType), _h(order) {}

    int getOrder() const { return _h.getOrder(); }

    BasisTypeEnum getBasisType() const { return _basisType; }

    void fillEvaluation(lsst::ndarray::Array<Pixel,1> const & array, double x, double y) const {
        _h.fillEvaluation(array, x, y);
        ConversionMatrix::convertOperationVector(array, HERMITE, _basisType, getOrder());
    }

    void fillEvaluation(lsst::ndarray::Array<Pixel,1> const & array, geom::Point2D const & point) const {
        _h.fillEvaluation(array, point);
        ConversionMatrix::convertOperationVector(array, HERMITE, _basisType, getOrder());
    }

    void fillEvaluation(lsst::ndarray::Array<Pixel,1> const & array, geom::Extent2D const & point) const {
        _h.fillEvaluation(array, point);
        ConversionMatrix::convertOperationVector(array, HERMITE, _basisType, getOrder());
    }

    void fillIntegration(lsst::ndarray::Array<Pixel,1> const & array, int xMoment=0, int yMoment=0) const {
        _h.fillIntegration(array, xMoment, yMoment);
        ConversionMatrix::convertOperationVector(array, HERMITE, _basisType, getOrder());
    }

private:
    BasisTypeEnum _basisType;
    HermiteEvaluator _h;
};

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_BASISEVALUATOR_H)
