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
#include "lsst/afw/detection/FootprintArray.h"
#include "lsst/afw/detection/FootprintArray.cc"
#include "ndarray/eigen.h"

namespace lsst { namespace afw { namespace math { namespace shapelets {

namespace {

static double const NORMALIZATION = std::pow(geom::PI, -0.25);

void fillCoordinates(
    detection::Footprint const & region, Eigen::ArrayXd & xArray, Eigen::ArrayXd & yArray
) {
    int n = 0;
    for (
        detection::Footprint::SpanList::const_iterator spanIter = region.getSpans().begin();
        spanIter != region.getSpans().end();
        ++spanIter
    ) {
        for (int x = (**spanIter).getX0(); x <= (**spanIter).getX1(); ++x, ++n) {
            xArray[n] = x;
            yArray[n] = (**spanIter).getY();
        }
    }
}

void fillHermite1d(Eigen::ArrayXXd & workspace, Eigen::ArrayXd const & coord) {
    if (workspace.cols() > 0)
        workspace.col(0) = NORMALIZATION * (-0.5 * coord.square()).exp();
    if (workspace.cols() > 1)
        workspace.col(1) = workspace.col(0) * std::sqrt(2.0);
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

template <typename ImagePixelT>
ModelBuilder::ModelBuilder(
    int order, BasisTypeEnum basisType,
    geom::ellipses::Ellipse const & ellipse,
    detection::Footprint const & region,
    image::Image<ImagePixelT> const & img
) : _order(order), _basisType(basisType), _region(region), _ellipse(ellipse) {
    _region.clipTo(img.getBBox(image::PARENT));
    _allocate();
    detection::flattenArray(_region, img.getArray(), _data, img.getXY0());
    fillCoordinates(_region, _x, _y);
}

template <typename ImagePixelT>
ModelBuilder::ModelBuilder(
    int order, BasisTypeEnum basisType,
    geom::ellipses::Ellipse const & ellipse,
    detection::Footprint const & region,
    image::MaskedImage<ImagePixelT> const & img,
    image::MaskPixel andMask,
    bool useVariance
) : _order(order), _basisType(basisType), _region(region), _ellipse(ellipse) {
    _region.intersectMask(*img.getMask(), andMask);
    _allocate();
    detection::flattenArray(_region, img.getImage()->getArray(), _data, img.getXY0());
    if (useVariance) {
        _weights = ndarray::allocate(_data.getSize<0>());
        detection::flattenArray(_region, img.getVariance()->getArray(), _weights, img.getXY0());
        _weights.asEigen<Eigen::ArrayXpr>() = _weights.asEigen<Eigen::ArrayXpr>().sqrt().inverse();
        _data.asEigen<Eigen::ArrayXpr>() *= _weights.asEigen<Eigen::ArrayXpr>();
    }
    fillCoordinates(_region, _x, _y);
}

void ModelBuilder::_allocate() {
    int nPix = _region.getArea();
    int nCoeff = computeSize(_order);
    _design = ndarray::allocate(nPix, nCoeff);
    _data = ndarray::allocate(nPix);
    // Note that we don't allocate weights here; we might not have any.
    _x.resize(nPix);
    _y.resize(nPix);
    _xt.resize(nPix);
    _yt.resize(nPix);
    _xWorkspace.resize(nPix, _order + 1);
    _yWorkspace.resize(nPix, _order + 1);
}

void ModelBuilder::update(geom::ellipses::Ellipse const & ellipse) {
    typedef geom::AffineTransform AT;
    _ellipse = ellipse;
    AT transform = _ellipse.getGridTransform();
    _xt = _x * transform[AT::XX] + _y * transform[AT::XY] + transform[AT::X];
    _yt = _x * transform[AT::YX] + _y * transform[AT::YY] + transform[AT::Y];
    fillHermite1d(_xWorkspace, _xt);
    fillHermite1d(_yWorkspace, _yt);
    ndarray::EigenView<Pixel,2,-2,Eigen::ArrayXpr> design(_design);
    for (PackedIndex i; i.getOrder() <= _order; ++i) {
        design.col(i.getIndex()) = _xWorkspace.col(i.getX()) * _yWorkspace.col(i.getY());
    }
    if (!_weights.isEmpty()) {
        for (int n = 0; n < design.cols(); ++n) {
            design.col(n) *= _weights.asEigen<Eigen::ArrayXpr>();
        }
    }
}

template <typename ImagePixelT>
void ModelBuilder::addToImage(
    image::Image<ImagePixelT> & img,
    ndarray::Array<Pixel const,1,1> const & coefficients,
    bool useWeights
) const {
    int n = 0;
    ndarray::EigenView<Pixel,2,-2> design(_design);
    ndarray::EigenView<Pixel const,1,1> coeff(coefficients);
    for (
        detection::Footprint::SpanList::const_iterator spanIter = _region.getSpans().begin();
        spanIter != _region.getSpans().end();
        ++spanIter
    ) {
        typename image::Image<ImagePixelT>::x_iterator pixIter
            = img.x_at((**spanIter).getX0(), (**spanIter).getY());
        for (int x = (**spanIter).getX0(); x <= (**spanIter).getX1(); ++x, ++pixIter, ++n) {
            Pixel v = design.row(n).dot(coeff);
            if (!useWeights && !_weights.isEmpty()) {
                v /= _weights[n];  // if _weights is not empty, design matrix already includes weights
            }
            *pixIter += v;
        }
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
    Eigen::Matrix<Pixel,6,Eigen::Dynamic> finalJac = gtJac * jacobian.transpose();
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
                block.row(n) += jacobian(AT::YY, n) * _y * tmp;
            if (std::abs(jacobian(AT::Y, n)) > eps)
                block.row(n) += jacobian(AT::Y, n) * tmp;
        }
    }
}


template ModelBuilder::ModelBuilder(
    int, BasisTypeEnum, geom::ellipses::Ellipse const &, detection::Footprint const &,
    image::Image<float> const &
);

template ModelBuilder::ModelBuilder(
    int, BasisTypeEnum, geom::ellipses::Ellipse const &, detection::Footprint const &,
    image::Image<double> const &
);

template ModelBuilder::ModelBuilder(
    int, BasisTypeEnum, geom::ellipses::Ellipse const &, detection::Footprint const &,
    image::MaskedImage<float> const &, image::MaskPixel, bool
);

template ModelBuilder::ModelBuilder(
    int, BasisTypeEnum, geom::ellipses::Ellipse const &, detection::Footprint const &,
    image::MaskedImage<double> const &, image::MaskPixel, bool
);

template void ModelBuilder::addToImage(
    image::Image<float> & img,
    ndarray::Array<Pixel const,1,1> const & coefficients,
    bool useWeights
) const;

template void ModelBuilder::addToImage(
    image::Image<double> & img,
    ndarray::Array<Pixel const,1,1> const & coefficients,
    bool useWeights
) const;

}}}} // namespace lsst::afw::math::shapelets
