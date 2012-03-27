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
#include "lsst/afw/math/shapelets/BasisEvaluator.h"
#include "lsst/afw/detection/FootprintFunctor.h"
#include "ndarray/eigen.h"


namespace lsst { namespace afw { namespace math { namespace shapelets {

namespace {

static inline void validateSize(int expected, int actual) {
    if (expected != actual) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthErrorException,
            (boost::format(
                "Output array for BasisEvaluator has incorrect size (%n, should be %n)."
            ) % actual % expected).str()
        );
    }
}

template <typename MaskedImageT>
class ShapeletLeastSquaresFunctor : public detection::FootprintFunctor<MaskedImageT> {
public:

    explicit ShapeletLeastSquaresFunctor(
        BasisEvaluator const * evaluator,
        ndarray::Array<Pixel,2,1> const & matrix,
        ndarray::Array<Pixel,1,1> const & vector,
        MaskedImageT const & data,
        geom::ellipses::Ellipse const & ellipse,
        image::MaskPixel andMask
    ) :
        detection::FootprintFunctor<MaskedImageT>(data),
        _evaluator(evaluator), _andMask(andMask), 
        _transform(ellipse.getGridTransform()),
        _matrix(matrix), _vector(vector),
        _workspace(ndarray::allocate(vector.getSize<0>()))
        {}

    void operator()(typename MaskedImageT::xy_locator loc, int x, int y) {
        if (this->getImage().getMask(true) && (loc.mask() & _andMask)) return;
        Pixel weight = 1.0;
        if (this->getImage().getVariance(true)) weight = 1.0 / loc.variance();
        geom::Point2D p = _transform(geom::Point2D(x, y));
        _evaluator->fillEvaluation(_workspace.shallow(), p);
        Eigen::SelfAdjointView<ndarray::EigenView<Pixel,2,1>,Eigen::Lower>(_matrix)
            .rankUpdate(_workspace, weight);
        _vector += loc.image() * weight * _workspace;
    }

private:
    BasisEvaluator const * _evaluator;
    image::MaskPixel _andMask;
    geom::AffineTransform _transform;
    ndarray::EigenView<Pixel,2,1> _matrix;
    ndarray::EigenView<Pixel,1,1> _vector;
    ndarray::EigenView<Pixel,1,1> _workspace;
};

} // anonymous

void BasisEvaluator::fillEvaluation(
    ndarray::Array<Pixel,1> const & array, double x, double y,
    ndarray::Array<Pixel,1> const & dx,
    ndarray::Array<Pixel,1> const & dy
) const {
    validateSize(computeSize(getOrder()), array.getSize<0>());
    _h.fillEvaluation(array, x, y, dx, dy);
    ConversionMatrix::convertOperationVector(array, HERMITE, _basisType, getOrder());
    if (!dx.isEmpty()) {
        validateSize(computeSize(getOrder()), dx.getSize<0>());
        ConversionMatrix::convertOperationVector(dx, HERMITE, _basisType, getOrder());
    }
    if (!dy.isEmpty()) {
        validateSize(computeSize(getOrder()), dy.getSize<0>());
        ConversionMatrix::convertOperationVector(dy, HERMITE, _basisType, getOrder());
    }
}

void BasisEvaluator::fillIntegration(ndarray::Array<Pixel,1> const & array, int xMoment, int yMoment) const {
    validateSize(computeSize(getOrder()), array.getSize<0>());
    _h.fillIntegration(array, xMoment, yMoment);
    ConversionMatrix::convertOperationVector(array, HERMITE, _basisType, getOrder());
}

template <typename ImagePixelT>
void BasisEvaluator::fillLeastSquares(
    ndarray::Array<Pixel,2,1> const & matrix,
    ndarray::Array<Pixel,1,1> const & vector,
    image::Image<ImagePixelT> const & data,
    detection::Footprint const & region,
    geom::ellipses::Ellipse const & ellipse
) const {
    image::MaskedImage<ImagePixelT> mi(boost::make_shared< image::Image<ImagePixelT> >(data));
    fillLeastSquares(matrix, vector, mi, region, ellipse);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void BasisEvaluator::fillLeastSquares(
    ndarray::Array<Pixel,2,1> const & matrix,
    ndarray::Array<Pixel,1,1> const & vector,
    image::MaskedImage<ImagePixelT,MaskPixelT,VariancePixelT> const & data,
    detection::Footprint const & region,
    geom::ellipses::Ellipse const & ellipse,
    image::MaskPixel andMask
) const {
    int const size = computeSize(getOrder());
    validateSize(size, matrix.getSize<0>());
    validateSize(size, matrix.getSize<1>());
    validateSize(size, vector.getSize<0>());
    ShapeletLeastSquaresFunctor< image::MaskedImage<ImagePixelT,MaskPixelT,VariancePixelT> > functor(
        this, matrix, vector, data, ellipse, andMask
    );
    functor.apply(region);
}

template void BasisEvaluator::fillLeastSquares(
    ndarray::Array<Pixel,2,1> const & matrix,
    ndarray::Array<Pixel,1,1> const & vector,
    image::Image<float> const & data,
    detection::Footprint const & region,
    geom::ellipses::Ellipse const & ellipse
) const;

template void BasisEvaluator::fillLeastSquares(
    ndarray::Array<Pixel,2,1> const & matrix,
    ndarray::Array<Pixel,1,1> const & vector,
    image::Image<double> const & data,
    detection::Footprint const & region,
    geom::ellipses::Ellipse const & ellipse
) const;

template void BasisEvaluator::fillLeastSquares(
    ndarray::Array<Pixel,2,1> const & matrix,
    ndarray::Array<Pixel,1,1> const & vector,
    image::MaskedImage<float> const & data,
    detection::Footprint const & region,
    geom::ellipses::Ellipse const & ellipse,
    image::MaskPixel andMask
) const;

template void BasisEvaluator::fillLeastSquares(
    ndarray::Array<Pixel,2,1> const & matrix,
    ndarray::Array<Pixel,1,1> const & vector,
    image::MaskedImage<double> const & data,
    detection::Footprint const & region,
    geom::ellipses::Ellipse const & ellipse,
    image::MaskPixel andMask
) const;

}}}} // namespace lsst::afw::math::shapelets
