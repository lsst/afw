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

template <typename ImageT>
int countPixels(
    detection::Footprint const & region,
    ImageT const & img, image::MaskPixel andMask
) {
    int n = 0;
    for (
        detection::Footprint::SpanList::const_iterator i = region.getSpans().begin();
        i != region.getSpans().end();
        ++i
    ) {
        typename ImageT::x_iterator iter = img.x_at((**i).getX0(), (**i).getY());
        typename ImageT::x_iterator const end = img.x_at((**i).getX1() + 1, (**i).getY());
        for (; iter != end; ++iter) {
            if (!(iter.mask() & andMask)) ++n;
        }
    }
    return n;
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

} // anonymous

template <typename ImagePixelT>
ModelBuilder::ModelBuilder(
    int order, BasisTypeEnum basisType,
    geom::ellipses::Ellipse const & ellipse,
    detection::Footprint const & region,
    image::Image<ImagePixelT> const & img
) : _order(order), _basisType(basisType), _ellipse(ellipse) {
    _allocate(region.getArea());
    detection::flattenArray(region, img.getArray(), _data, img.getXY0());
    int n = 0;
    for (
        detection::Footprint::SpanList::const_iterator i = region.getSpans().begin();
        i != region.getSpans().end();
        ++i
    ) {
        typename image::Image<ImagePixelT>::x_iterator iter 
            = img.x_at((**i).getX0(), (**i).getY());
        for (int x = (**i).getX0(); x <= (**i).getX1(); ++iter, ++x) {
            _data[n] = *iter;
            _x[n] = x;
            _y[n] = (**i).getY();
            ++n;
        }
    }
}

template <typename ImagePixelT>
ModelBuilder::ModelBuilder(
    int order, BasisTypeEnum basisType,
    geom::ellipses::Ellipse const & ellipse,
    detection::Footprint const & region,
    image::MaskedImage<ImagePixelT> const & img,
    image::MaskPixel andMask,
    bool useVariance
) : _order(order), _basisType(basisType), _ellipse(ellipse) {
    _allocate(countPixels(region, img, andMask));
    if (useVariance) _weights = ndarray::allocate(_data.getSize<0>());
    int n = 0;
    for (
        detection::Footprint::SpanList::const_iterator i = region.getSpans().begin();
        i != region.getSpans().end();
        ++i
    ) {
        typename image::MaskedImage<ImagePixelT>::x_iterator iter 
            = img.x_at((**i).getX0(), (**i).getY());
        for (int x = (**i).getX0(); x <= (**i).getX1(); ++iter, ++x) {
            if (!(iter.mask() & andMask)) {
                _data[n] = iter.image();
                if (useVariance) {
                    _weights[n] = 1.0 / iter.variance();
                    _data[n] *= _weights[n];
                }
                _x[n] = x;
                _y[n] = (**i).getY();
                ++n;
            }
        }
    }
}

void ModelBuilder::_allocate(int nPix) {
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
    if (!_weights.isEmpty()) design *= _weights.asEigen<Eigen::ArrayXpr>();
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

}}}} // namespace lsst::afw::math::shapelets
