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

/**
 *  @brief A class that evaluates a Gauss-Hermite (shapelet with HERMITE basis type) basis over a footprint.
 *
 *  This is intended as the primary way to build shapelet models that will be fit to image data.  Given
 *  an Image or MaskedImage, it flattens the pixels according to a footprint and generates a
 *  (pixels) x (basis function) matrix that can be fit to the flattened data vector with linear
 *  least-squares.  It can also produce the derivative of this matrix w.r.t. the ellipse parameters.
 *
 *  Unlike virtually everything else in the shapelets library, ModelBuilder does not rely on the
 *  HermiteEvaluator class to compute basis functions.  Instead of making the iteration over the
 *  pixels the outer loop, it uses Eigen array objects that are the size of an entire image.  This
 *  uses more memory for temporaries, but it takes advantage of Eigen's vectorized arithmetic operators.
 */
class ModelBuilder {
public:

    /**
     *  @brief Construct a ModelBuilder that can be used to fit data from an Image.
     *
     *  @param[in] order       Order of the shapelet model.
     *  @param[in] ellipse     Basis ellipse for the shapelet model.  This can be
     *                         changed after construction, but the parameterization
     *                         of the ellipse used in the definition of derivatives
     *                         is based on the ellipse the ModelBuilder was
     *                         constructed with.
     *  @param[in] region      Footprint that defines where on the image to evaluate
     *                         the model.  The footprint need not be contained by
     *                         the image's bounding box.
     *  @param[in] img         Image whose pixels will be flattened into the data
     *                         array.
     */
    template <typename ImagePixelT>
    explicit ModelBuilder(
        int order,
        geom::ellipses::Ellipse const & ellipse,
        detection::Footprint const & region,
        image::Image<ImagePixelT> const & img
    );

    /**
     *  @brief Construct a ModelBuilder that can be used to fit data from an Image.
     *
     *  @param[in] order       Order of the shapelet model.
     *  @param[in] ellipse     Basis ellipse for the shapelet model.  This can be
     *                         changed after construction, but the parameterization
     *                         of the ellipse used in the definition of derivatives
     *                         is based on the ellipse the ModelBuilder was
     *                         constructed with.
     *  @param[in] region      Footprint that defines where on the image to evaluate
     *                         the model.  The footprint need not be contained by the
     *                         image's bounding box.
     *  @param[in] img         MaskedImage whose pixels will be flattened into the data
     *                         array, and whose mask and variance planes may be used
     *                         to reject pixels and set the weights, respectively.
     *  @param[in] andMask     Bitmask that will be used to ignore pixels by removing
     *                         them from the region footprint before using it to
     *                         flatten the data pixels (and possibly variance pixels).
     *  @param[in] useVariance If true, the design matrix and data vector will be
     *                         multiplied by the inverse variance.
     */
    template <typename ImagePixelT>
    explicit ModelBuilder(
        int order, 
        geom::ellipses::Ellipse const & ellipse,
        detection::Footprint const & region,
        image::MaskedImage<ImagePixelT> const & img,
        image::MaskPixel andMask=0x0,
        bool useVariance=true
    );

    /**
     *  @brief Update the basis ellipse and recompute the design matrix.
     *
     *  This does not change the ellipse parameterization used by computeDerivative.
     */
    void update(geom::ellipses::Ellipse const & ellipse);

    /// @brief Return the design matrix (may or may not be weighted, depending on construction).
    ndarray::Array<Pixel const,2,-2> getDesignMatrix() const { return _design; }

    /// @brief Return the data vector (may or may not be weighted, depending on construction).
    ndarray::Array<Pixel const,1,1> getDataVector() const { return _data; }

    /**
     *  @brief Return the region that defines the pixels in the model.
     *
     *  This may differ from the footprint supplied at construction if the andMask argument
     *  was used to reject bad pixels from a MaskedImage.
     */
    detection::Footprint const & getRegion() const { return _region; }
    
    /**
     *  @brief Add the model to an image.
     *
     *  @param[in,out]  img           Image the model will be added to.  Only pixels in the region
     *                                footprint will be affected.
     *  @param[in]      coefficients  Shapelet coefficients for the model.
     *  @param[in]      useWeights    If true, the model will include the weights (if there are any).
     *
     *  Note that the model can be subtracted instead simply by negating the coefficient array.
     */
    template <typename ImagePixelT>
    void addToImage(
        image::Image<ImagePixelT> & img,
        ndarray::Array<Pixel const,1,1> const & coefficients,
        bool useWeights = false
    ) const;

    /**
     *  @brief Evaluate the derivative of the model with respect to the ellipse parameters
     *         or a function thereof.
     *
     *  @param[out]   output       Array that will contain the derivative.  The dimensions
     *                             are ordered {data points, basis elements, ellipse parameters}.
     *                             Must be preallocated to the correct dimensions.
     */
    void computeDerivative(ndarray::Array<Pixel,3,-3> const & output) const;

    /**
     *  @brief Evaluate the derivative of the model with respect to the ellipse parameters
     *         or a function thereof.
     *
     *  @param[out]   output       Array that will contain the derivative.  The dimensions
     *                             are ordered {data points, basis elements, ellipse parameters}.
     *                             Must be preallocated to the correct dimensions.
     *  @param[in]    jacobian     Matrix giving the partial derivatives of the ellipse parameters
     *                             with respect to the desired parameters.  Each row corresponds
     *                             to a single ellipse parameter, and each column corresponds
     *                             to a desired output parameter.
     *  @param[in]    add          If true, the derivative will be added to the output array
     *                             instead of overwriting it.
     *
     *  Passing a Jacobian matrix to computeDerivative is equivalent to multiplying the output
     *  by the Jacobian on the right; that is:
     *  @code
     *  computeDerivative(output, jacobian);
     *  @endcode
     *  is equivalent to
     *  @code
     *  computeDerivative(tmp);
     *  for (int n = 0; n < tmp.getSize<0>(); ++n)
     *      output[n].asEigen() = tmp[n].asEigen() * jacobian;
     *  @endcode
     *  The second may be significantly slower, however, because the tmp tensor is never
     *  actually formed in the first case.
     */
    void computeDerivative(
        ndarray::Array<Pixel,3,-3> const & output,
        Eigen::Matrix<Pixel,5,Eigen::Dynamic> const & jacobian,
        bool add=false
    ) const;

private:

    // Implemenatation takes Jacobian wrt affine transform parameters instead of ellipse parameters.
    void _computeDerivative(
        ndarray::Array<Pixel,3,-3> const & output,
        Eigen::Matrix<Pixel,6,Eigen::Dynamic> const & jacobian,
        bool add
    ) const;

    void _allocate();

    int _order;
    detection::Footprint _region;
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
};

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_ModelBuilder_h_INCLUDED)
