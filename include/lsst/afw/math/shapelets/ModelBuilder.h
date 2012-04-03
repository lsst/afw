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

    /**
     *  @brief Construct a ModelBuilder that can be used to fit data from an Image.
     *
     *  @param[in] order       Order of the shapelet model.
     *  @param[in] basisType   Basis type of the shapelet model.
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
        int order, BasisTypeEnum basisType,
        geom::ellipses::Ellipse const & ellipse,
        detection::Footprint const & region,
        image::Image<ImagePixelT> const & img
    );

    /**
     *  @brief Construct a ModelBuilder that can be used to fit data from an Image.
     *
     *  @param[in] order       Order of the shapelet model.
     *  @param[in] basisType   Basis type of the shapelet model.
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
        int order, BasisTypeEnum basisType,
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
    void addModelToImage(
        image::Image<ImagePixelT> & img,
        ndarray::Array<Pixel const,1,1> const & coefficients,
        bool useWeights = false
    );

    void computeDerivative(
        ndarray::Array<Pixel,3> const & output,
        Eigen::Matrix<Pixel,Eigen::Dynamic,5> const & jacobian
    ) const;

private:

    void _allocate();

    int _order;
    BasisTypeEnum _basisType;
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
    Eigen::ArrayXXd _dxWorkspace;
    Eigen::ArrayXXd _dyWorkspace;
};

}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_ModelBuilder_h_INCLUDED)
