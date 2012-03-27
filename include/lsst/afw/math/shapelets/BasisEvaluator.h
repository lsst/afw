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

    typedef boost::shared_ptr<BasisEvaluator> Ptr;
    typedef boost::shared_ptr<BasisEvaluator const> ConstPtr;

    /// @brief Construct an evaluator for the given function.
    BasisEvaluator(int order, BasisTypeEnum basisType) : _basisType(basisType), _h(order) {}

    int getOrder() const { return _h.getOrder(); }

    BasisTypeEnum getBasisType() const { return _basisType; }

    void fillEvaluation(
        ndarray::Array<Pixel,1> const & array, double x, double y,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const;

    void fillEvaluation(
        ndarray::Array<Pixel,1> const & array, geom::Point2D const & point,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const {
        fillEvaluation(array, point.getX(), point.getY(), dx, dy);
    }

    void fillEvaluation(
        ndarray::Array<Pixel,1> const & array, geom::Extent2D const & point,
        ndarray::Array<Pixel,1> const & dx = ndarray::Array<Pixel,1>(),
        ndarray::Array<Pixel,1> const & dy = ndarray::Array<Pixel,1>()
    ) const {
        fillEvaluation(array, point.getX(), point.getY(), dx, dy);
    }

    void fillIntegration(ndarray::Array<Pixel,1> const & array, int xMoment=0, int yMoment=0) const;

    /**
     *  @brief Fill the matrix and vector that define the normal equations used in solving
     *         a linear least squares problem.
     *
     *  If @f$M@f$ is the design matrix that maps shapelet coefficients to pixel values, and @f$d@f$
     *  is the flattened data vector, then the output matrix is equal to @f$M^T M@f$ and the output
     *  vector is equal to @F$M^T d@f$.  The design matrix itself is never formed.
     *
     *  The outputs will be added to, not overwritten, so they should be initialized to zero if
     *  desired (this allows this function to be called repeatedly to construct a least-squares
     *  equation that fits the same model to multiple images).
     *
     *  @param[out]  matrix   A square matrix with dimensions corresponding to the size of the basis.
     *  @param[out]  vector   A vector with dimensions corresponding to the size of the basis.
     *  @param[in]   data     Image containing pixel values to be fit.
     *  @param[in]   region   A Footprint that defines the region of the image to be fit.
     *  @param[in]   ellipse  Basis ellipse for the shapelet model.
     *
     *  Note that the ellipse transformation is applied without correcting for its determinant;
     *  the total flux of the model is a function of the area of the ellipse (and the coefficients,
     *  of course).
     */
    template <typename ImagePixelT>
    void fillLeastSquares(
        ndarray::Array<Pixel,2,1> const & matrix,
        ndarray::Array<Pixel,1,1> const & vector,
        image::Image<ImagePixelT> const & data,
        detection::Footprint const & region,
        geom::ellipses::Ellipse const & ellipse
    ) const;

    /**
     *  @brief Fill the matrix and vector that define the normal equations used in solving
     *         a weighted linear least squares problem.
     *
     *  If @f$M@f$ is the design matrix that maps shapelet coefficients to pixel values, @f$d@f$
     *  is the flattened data vector, and @f$V@f$ is a diagonal matrix with containing the pixel
     *  variances, then the output matrix is equal to @f$M^T V^{-1} M@f$ and the output vector
     *  is equal to @F$M^T V^{-1} d@f$.  The design matrix itself is never formed.
     *
     *  The outputs will be added to, not overwritten, so they should be initialized to zero if
     *  desired (this allows this function to be called repeatedly to construct a least-squares
     *  equation that fits the same model to multiple images).
     *
     *  @param[out]  matrix   A square matrix with dimensions corresponding to the size of the basis.
     *  @param[out]  vector   A vector with dimensions corresponding to the size of the basis.
     *  @param[in]   data     MaskedImage containing pixel values to be fit, and variances to be used
     *                        as weights.  Pixels with mask values corresponding to andMask will be
     *                        ignored.
     *  @param[in]   region   A Footprint that defines the region of the image to be fit.
     *  @param[in]   ellipse  Basis ellipse for the shapelet model.
     *  @param[in]   andMask  Defines which mask bits cause a value to be ignored.
     *
     *  Note that the ellipse transformation is applied without correcting for its determinant;
     *  the total flux of the model is a function of the area of the ellipse (and the coefficients,
     *  of course).
     */
    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
    void fillLeastSquares(
        ndarray::Array<Pixel,2,1> const & matrix,
        ndarray::Array<Pixel,1,1> const & vector,
        image::MaskedImage<ImagePixelT,MaskPixelT,VariancePixelT> const & data,
        detection::Footprint const & region,
        geom::ellipses::Ellipse const & ellipse,
        image::MaskPixel andMask=0x0
    ) const;

private:
    BasisTypeEnum _basisType;
    HermiteEvaluator _h;
};

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_BASISEVALUATOR_H)
