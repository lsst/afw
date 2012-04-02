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

#include "lsst/afw/math/shapelets/HermiteEvaluator.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace math { namespace shapelets {

namespace {

static double const NORMALIZATION = std::pow(geom::PI, -0.25);

/**
 *  @brief An iterator-like construct for evaluating either normalized Hermite polynomials or
 *         Gauss-Hermite functions at a point.
 */
class HermiteRecurrenceRelation {
public:

    static int const computeSize(int order) { return order + 1; }

    /// @brief Return the value of the shapelet basis function at the current order.
    double operator()() const { return _current; }

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

void fillEvaluation1d(
    ndarray::Array<Pixel,1,1> const & result,
    double x,
    ndarray::Array<Pixel,1,1> const & dx = ndarray::Array<Pixel,1,1>()
) {
    HermiteRecurrenceRelation r(x, NORMALIZATION * std::exp(-0.5*x*x));
    ndarray::Array<Pixel,1,1>::Iterator const end = result.end();
    for (ndarray::Array<Pixel,1,1>::Iterator i = result.begin(); i != end; ++i, ++r) {
        *i = r();
    }
    if (!dx.isEmpty() && dx.getSize<0>() > 0) {
        assert(dx.getSize<0>() == result.getSize<0>());
        dx[0] = -x * result[0];
        for (int n = 1; n < result.getSize<0>(); ++n) {
            dx[n] = -x * result[n] + std::sqrt(2.0 * n) * result[n - 1];
        }
    }
}

void fillIntegration1d(ndarray::Array<Pixel,1,1> const & result, int moment) {
    int const order = result.getSize<0>() - 1;
    result.deep() = 0.0;
    result[0] = std::pow(4.0*geom::PI, 0.25);
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

Eigen::MatrixXd computeInnerProductMatrix1d(int rowOrder, int colOrder, double a, double b) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rowOrder + 1, colOrder + 1);
    double v = 1.0 / (a*a + b*b);
    double f1 = 2.0 * a * b * v;
    double f2 = (a*a - b*b) * v;
    result(0, 0) = std::sqrt(2.0 * v);
    for (int j = 2; j <= rowOrder; j += 2) {
        result(j, 0) = f2 * std::sqrt(double(j-1) / j) * result(j-2, 0);
    }
    for (int k = 2; k <= colOrder; k += 2) {
        result(0, k) = -f2 * std::sqrt(double(k-1) / k) * result(0, k-2);
    }
    if (rowOrder > 0 && colOrder > 0) result(1, 1) = f1 * result(0,0);
    if (colOrder > 0) {
        for (int j = 3; j <= rowOrder; j += 2) {
            result(j, 1) = f1 * std::sqrt(j) * result(j-1, 0);
        }
    }
    if (rowOrder > 0) {
        for (int k = 3; k <= colOrder; k += 2) {
            result(1, k) = f1 * std::sqrt(k) * result(0, k-1);
        }
    }
    for (int j = 2; j <= rowOrder; ++j) {
        double q2 = std::sqrt(double(j-1)/j);
        if (j <= colOrder) {
            result(j, j) = f1 * result(j-1, j-1) + f2 * q2 * result(j-2, j);
        }
        for (int k = (j%2) + 2; k <= colOrder; k += 2) {
            double q1 = std::sqrt(double(k) / j);
            result(j, k) = f1 * q1 * result(j-1, k-1) + f2 * q2 * result(j-2, k);
        }
    }
    return result;
}

void weaveFill(
    ndarray::Array<Pixel,1> const & target,
    ndarray::Array<Pixel const,1,1> const & x,
    ndarray::Array<Pixel const,1,1> const & y
) {
    int const order = x.getSize<0>() - 1;
    for (PackedIndex i; i.getOrder() <= order; ++i) {
        target[i.getIndex()] = x[i.getX()] * y[i.getY()];
    }
}

double weaveSum(
    ndarray::Array<Pixel const,1> const & target,
    ndarray::Array<Pixel const,1,1> const & x,
    ndarray::Array<Pixel const,1,1> const & y
) {
    double r = 0.0;
    int const order = x.getSize<0>() - 1;
    for (PackedIndex i; i.getOrder() <= order; ++i) {
        r += target[i.getIndex()] * x[i.getX()] * y[i.getY()];
    }
    return r;
}

} // anonymous    

void HermiteEvaluator::fillEvaluation(
    ndarray::Array<Pixel,1> const & target, double x, double y,
    ndarray::Array<Pixel,1> const & dx,
    ndarray::Array<Pixel,1> const & dy 
) const {
    if (dx.isEmpty()) {
        fillEvaluation1d(_xWorkspace, x);
    } else {
        fillEvaluation1d(_xWorkspace, x, _dxWorkspace);
    }
    if (dy.isEmpty()) {
        fillEvaluation1d(_yWorkspace, y);
    } else {
        fillEvaluation1d(_yWorkspace, y, _dyWorkspace);
    }
    weaveFill(target, _xWorkspace, _yWorkspace);
    if (!dx.isEmpty()) weaveFill(dx, _dxWorkspace, _yWorkspace);
    if (!dy.isEmpty()) weaveFill(dy, _xWorkspace, _dyWorkspace);
}

void HermiteEvaluator::fillIntegration(
    ndarray::Array<Pixel,1> const & target, int xMoment, int yMoment
) const {
    fillIntegration1d(_xWorkspace, xMoment);
    fillIntegration1d(_yWorkspace, yMoment);
    return weaveFill(target, _xWorkspace, _yWorkspace);
}

double HermiteEvaluator::sumEvaluation(
    ndarray::Array<Pixel const,1> const & coeff, double x, double y,
    double * dx, double * dy
) const {
    if (!dx) {
        fillEvaluation1d(_xWorkspace, x);
    } else {
        fillEvaluation1d(_xWorkspace, x, _dxWorkspace);
    }
    if (!dy) {
        fillEvaluation1d(_yWorkspace, y);
    } else {
        fillEvaluation1d(_yWorkspace, y, _dyWorkspace);
    }
    if (dx) *dx = weaveSum(coeff, _dxWorkspace, _yWorkspace);
    if (dy) *dy = weaveSum(coeff, _xWorkspace, _dyWorkspace);
    return weaveSum(coeff, _xWorkspace, _yWorkspace);
}

double HermiteEvaluator::sumIntegration(
    ndarray::Array<Pixel const,1> const & coeff, int xMoment, int yMoment
) const {
    fillIntegration1d(_xWorkspace, xMoment);
    fillIntegration1d(_yWorkspace, yMoment);
    return weaveSum(coeff, _xWorkspace, _yWorkspace);
}

HermiteEvaluator::HermiteEvaluator(int order) :
    _xWorkspace(ndarray::allocate(order + 1)),
    _yWorkspace(ndarray::allocate(order + 1)),
    _dxWorkspace(ndarray::allocate(order + 1)),
    _dyWorkspace(ndarray::allocate(order + 1))
{}

Eigen::MatrixXd HermiteEvaluator::computeInnerProductMatrix(
    int rowOrder, int colOrder, double a, double b
) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(computeSize(rowOrder), computeSize(colOrder));
    Eigen::MatrixXd m = computeInnerProductMatrix1d(rowOrder, colOrder, a, b);
    for (PackedIndex i; i.getOrder() <= rowOrder; ++i) {
        for (PackedIndex j; j.getOrder() <= colOrder; ++j) {
            result(i.getIndex(), j.getIndex()) = m(i.getX(), j.getX()) * m(i.getY(), j.getY());
        }
    }
    return result;
}

}}}} // namespace lsst::afw::math::shapelets
