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
 
#ifndef LSST_AFW_MATH_SHAPELETS_ModelBuilder_h_INCLUDED
#define LSST_AFW_MATH_SHAPELETS_ModelBuilder_h_INCLUDED

#include "lsst/afw/math/shapelets/BasisEvaluator.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

class ModelBuilder {
public:

    template <typename ImagePixelT>
    explicit ModelBuilder(
        int order, BasisTypeEnum basisType,
        geom::ellipses::Ellipse const & ellipse,
        detection::Footprint const & region,
        image::Image<ImagePixelT> const & img
    );

    template <typename ImagePixelT>
    explicit ModelBuilder(
        int order, BasisTypeEnum basisType,
        geom::ellipses::Ellipse const & ellipse,
        detection::Footprint const & region,
        image::MaskedImage<ImagePixelT> const & img,
        image::MaskPixel andMask=0x0,
        bool useVariance=true
    );

    void update(geom::ellipses::Ellipse const & ellipse);

    ndarray::Array<Pixel const,2,-2> getDesignMatrix() const { return _design; }

    ndarray::Array<Pixel const,1,1> getDataVector() const { return _data; }

    void computeDerivative(
        ndarray::Array<Pixel,3> const & output,
        Eigen::Matrix<Pixel,Eigen::Dynamic,5> const & jacobian
    ) const;

private:

    void _allocate(int nPix);

    int _order;
    BasisTypeEnum _basisType;
    geom::ellipses::Ellipse _ellipse;
    ndarray::Array<Pixel,2,-2> _design;
    ndarray::Array<Pixel,1,1> _data;
    ndarray::Array<Pixel,1,1> _weights;
    Eigen::ArrayXd _x;
    Eigen::ArrayXd _y;
    Eigen::ArrayXd _xt;
    Eigen::ArrayXd _yt;
    Eigen::ArrayXXd _xWorkspace;
    Eigen::ArrayXXd _yWorkspace;
    Eigen::ArrayXXd _dxWorkspace;
    Eigen::ArrayXXd _dyWorkspace;
};

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_ModelBuilder_h_INCLUDED)
