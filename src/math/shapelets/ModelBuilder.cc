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

#include "boost/format.hpp"
#include "boost/make_shared.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/shapelets/ModelBuilder.h"
#include "ndarray/eigen.h"

namespace lsst { namespace afw { namespace math { namespace shapelets {

namespace {

static double const NORMALIZATION = std::pow(geom::PI, -0.25);

void fillHermite1d(Eigen::ArrayXXd & workspace, Eigen::ArrayXd const & coord) {
    if (workspace.cols() > 0)
        workspace.col(0) = NORMALIZATION * (-0.5 * coord.square()).exp();
    if (workspace.cols() > 1)
        workspace.col(1) = std::sqrt(2.0) * coord * workspace.col(0);
    for (int j = 2; j < workspace.cols(); ++j) {
        workspace.col(j) = std::sqrt(2.0 / j) * coord * workspace.col(j-1)
            - std::sqrt((j - 1.0) / j) * workspace.col(j-2);
    }
}

void fillDerivative1d(
    Eigen::ArrayXXd & dWorkspace, Eigen::ArrayXXd const & workspace, Eigen::ArrayXd const & coord
) {
    dWorkspace.resize(workspace.rows(), workspace.cols());
    if (dWorkspace.cols() > 0)
        dWorkspace.col(0) = - coord * workspace.col(0);
    for (int j = 1; j < dWorkspace.cols(); ++j) {
        dWorkspace.col(j) = std::sqrt(2.0 * j) * workspace.col(j-1) - coord * workspace.col(j);
    }
}

} // anonymous

ModelBuilder::ModelBuilder(
    int order,
    geom::ellipses::Ellipse const & ellipse,
    detection::Footprint const & region
) : _order(order), _ellipse(ellipse),
    _model(ndarray::allocate(region.getArea(), computeSize(order))),
    _x(region.getArea()), _y(region.getArea()),
    _xt(region.getArea()), _yt(region.getArea()),
    _xWorkspace(region.getArea(), order + 1), _yWorkspace(region.getArea(), order + 1)
{
    int n = 0;
    for (
        detection::Footprint::SpanList::const_iterator i = region.getSpans().begin();
        i != region.getSpans().end();
        ++i
    ) {
        for (int x = (**i).getX0(); x <= (**i).getX1(); ++x, ++n) {
            _x[n] = x;
            _y[n] = (**i).getY();
        }
    }
    update(ellipse);
}

ModelBuilder::ModelBuilder(
    int order,
    geom::ellipses::Ellipse const & ellipse,
    afw::geom::Box2I const & region
) : _order(order), _ellipse(ellipse),
    _model(ndarray::allocate(region.getArea(), computeSize(order))),
    _x(region.getArea()), _y(region.getArea()),
    _xt(region.getArea()), _yt(region.getArea()),
    _xWorkspace(region.getArea(), order + 1), _yWorkspace(region.getArea(), order + 1)
{
    int n = 0;
    afw::geom::Point2I const llc = region.getMin();
    afw::geom::Point2I const urc = region.getMax();
    for (int y = llc.getY(); y <= urc.getY(); ++y) {
        for (int x = llc.getX(); x <= urc.getX(); ++x, ++n) {
            _x[n] = x;
            _y[n] = y;
        }
    }
    update(ellipse);
}


void ModelBuilder::update(geom::ellipses::Ellipse const & ellipse) {
    typedef geom::AffineTransform AT;
    _ellipse = ellipse;
    AT transform = _ellipse.getGridTransform();
    _xt = _x * transform[AT::XX] + _y * transform[AT::XY] + transform[AT::X];
    _yt = _x * transform[AT::YX] + _y * transform[AT::YY] + transform[AT::Y];
    fillHermite1d(_xWorkspace, _xt);
    fillHermite1d(_yWorkspace, _yt);
    ndarray::EigenView<Pixel,2,-2,Eigen::ArrayXpr> model(_model);
    for (PackedIndex i; i.getOrder() <= _order; ++i) {
        model.col(i.getIndex()) = _xWorkspace.col(i.getX()) * _yWorkspace.col(i.getY());
    }
}

void ModelBuilder::computeDerivative(ndarray::Array<Pixel,3,-3> const & output) const {
    Eigen::Matrix<Pixel,6,Eigen::Dynamic> gtJac(6, 5);
    gtJac.block<6,5>(0,0) = _ellipse.getGridTransform().d();
    _computeDerivative(output, gtJac, false);
}

void ModelBuilder::computeDerivative(
    ndarray::Array<Pixel,3,-3> const & output,
    Eigen::Matrix<Pixel,5,Eigen::Dynamic> const & jacobian,
    bool add
) const {
    geom::ellipses::Ellipse::GridTransform::DerivativeMatrix gtJac = _ellipse.getGridTransform().d();
    Eigen::Matrix<Pixel,6,Eigen::Dynamic> finalJac = gtJac * jacobian;
    _computeDerivative(output, finalJac, add);
}

void ModelBuilder::_computeDerivative(
    ndarray::Array<Pixel,3,-3> const & output,
    Eigen::Matrix<Pixel,6,Eigen::Dynamic> const & jacobian,
    bool add
) const {
    static double const eps = std::numeric_limits<double>::epsilon();
    typedef geom::AffineTransform AT;
    Eigen::ArrayXXd dxWorkspace(_xWorkspace.rows(), _xWorkspace.cols());
    Eigen::ArrayXXd dyWorkspace(_yWorkspace.rows(), _yWorkspace.cols());
    Eigen::ArrayXd tmp(_x.size());
    fillDerivative1d(dxWorkspace, _xWorkspace, _xt);
    fillDerivative1d(dyWorkspace, _yWorkspace, _yt);
    for (PackedIndex i; i.getOrder() <= _order; ++i) {
        ndarray::EigenView<Pixel,2,-1,Eigen::ArrayXpr> block(output[ndarray::view()(i.getIndex())()]);
        if (!add) block.setZero();
        // We expect the Jacobian to be pretty sparse, so instead of just doing
        // standard multiplications here, we inspect each element and only do the
        // products we'll need.
        // This way if the user wants, for example, the derivatives with respect
        // to the ellipticity, we don't waste time computing elements that are
        // only useful when computing the derivatives wrt the centroid.
        tmp = dxWorkspace.col(i.getX()) * _yWorkspace.col(i.getY());
        for (int n = 0; n < jacobian.cols(); ++n) {
            if (std::abs(jacobian(AT::XX, n)) > eps)
                block.col(n) += jacobian(AT::XX, n) * _x * tmp;
            if (std::abs(jacobian(AT::XY, n)) > eps)
                block.col(n) += jacobian(AT::XY, n) * _y * tmp;
            if (std::abs(jacobian(AT::X, n)) > eps)
                block.col(n) += jacobian(AT::X, n) * tmp;
        }
        tmp = _xWorkspace.col(i.getX()) * dyWorkspace.col(i.getY());
        for (int n = 0; n < jacobian.cols(); ++n) {
            if (std::abs(jacobian(AT::YX, n)) > eps)
                block.col(n) += jacobian(AT::YX, n) * _x * tmp;
            if (std::abs(jacobian(AT::YY, n)) > eps)
                block.col(n) += jacobian(AT::YY, n) * _y * tmp;
            if (std::abs(jacobian(AT::Y, n)) > eps)
                block.col(n) += jacobian(AT::Y, n) * tmp;
        }
    }
}


}}}} // namespace lsst::afw::math::shapelets
