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
 
#ifndef LSST_AFW_MATH_SHAPELETS_DETAIL_HERMITEEVALUATOR_H
#define LSST_AFW_MATH_SHAPELETS_DETAIL_HERMITEEVALUATOR_H

/**
 * @file
 *
 * @brief HermiteEvaluator, a utility class for evaluating HERMITE shapelet functions.
 *
 * @todo
 *
 * @author Jim Bosch
 */

#include "ndarray.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/shapelets/constants.h"
#include "Eigen/Core"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {
namespace detail {

/**
 *  @brief A class to evaluate HERMITE shapelet-related quantities.
 */
class HermiteEvaluator {
public:

    /**
     *  @brief Fill a matrix with the function inner products of two
     *         HERMITE shapelet basis functions with different scales. 
     *  @f$
     *      M_{\mathbf{i},\mathbf{j}} = 
     *      \int d^2 \mathbf{x} \psi_\mathbf{i}(a\mathbf{x})\phi_\mathbf{j}(b\mathbf{x})
     *  @f$
     */
    static Eigen::MatrixXd computeInnerProductMatrix(
        int rowOrder, int colOrder, double a, double b
    );

    int getOrder() const { return _xWorkspace.getSize<0>() - 1; }

    /**
     *  @brief Fill a vector whose dot product with a HERMITE coefficient vector evaluates a
     *         simple unscaled shapelet expansion at the given point.
     */
    void fillEvaluation(
        ndarray::Array<Pixel,1> const & target, double x, double y,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const;

    /**
     *  @brief Fill a vector whose dot product with a HERMITE coefficient vector evaluates a
     *         simple unscaled shapelet expansion at the given point.
     */
    void fillEvaluation(
        ndarray::Array<Pixel,1> const & target, geom::Point2D const & point,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const {
        fillEvaluation(target, point.getX(), point.getY(), dx, dy);
    }

    /**
     *  @brief Fill a vector whose dot product with a HERMITE coefficient vector evaluates a
     *         simple unscaled shapelet expansion at the given point.
     */
    void fillEvaluation(
        ndarray::Array<Pixel,1> const & target, geom::Extent2D const & point,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const {
        fillEvaluation(target, point.getX(), point.getY(), dx, dy);
    }    

    /**
     *  @brief Fill a vector whose dot product with a HERMITE coefficient vector integrates
     *         a simple unscaled shapelet expansion.
     */
    void fillIntegration(ndarray::Array<Pixel,1> const & target, int xMoment=0, int yMoment=0) const;

    /**
     *  @brief Evaluate a simple unscaled shapelet expansion at the given point.
     */
    double sumEvaluation(
        ndarray::Array<Pixel const,1> const & coeff, double x, double y,
        double * dx = 0, double * dy = 0
    ) const;

    /**
     *  @brief Evaluate a simple unscaled shapelet expansion at the given point.
     */
    double sumEvaluation(
        ndarray::Array<Pixel const,1> const & coeff, geom::Point2D const & point,
        double * dx = 0, double * dy = 0
    ) const {
        return sumEvaluation(coeff, point.getX(), point.getY(), dx, dy);
    }

    /**
     *  @brief Evaluate a simple unscaled shapelet expansion at the given point.
     */
    double sumEvaluation(
        ndarray::Array<Pixel const,1> const & coeff, geom::Extent2D const & point,
        double * dx = 0, double * dy = 0
    ) const {
        return sumEvaluation(coeff, point.getX(), point.getY(), dx, dy);
    }

    /**
     *  @brief Integrate a simple unscale shapelet expansion at the given point.
     */
    double sumIntegration(ndarray::Array<Pixel const,1> const & coeff, int xMoment=0, int yMoment=0) const;

    explicit HermiteEvaluator(int order);

private:

    ndarray::Array<Pixel,1,1> _xWorkspace;
    ndarray::Array<Pixel,1,1> _yWorkspace;
    ndarray::Array<Pixel,1,1> _dxWorkspace;
    ndarray::Array<Pixel,1,1> _dyWorkspace;
};

}}}}}   // lsst::afw::math::shapelets::detail

#endif // !defined(LSST_AFW_MATH_SHAPELETS_DETAIL_HERMITEEVALUATOR_H)
