// -*- LSST-C++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
#ifndef LSST_AFW_DETECTION_SHAPELETLOCALPSF_H
#define LSST_AFW_DETECTION_SHAPELETLOCALPSF_H

#include "lsst/afw/detection/LocalPsf.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace detection {

class ShapeletLocalPsf : public LocalPsf{
public: 
    typedef boost::shared_ptr<ShapeletLocalPsf> Ptr;
    typedef boost::shared_ptr<ShapeletLocalPsf const> ConstPtr;

    /**
     *  @brief Return true if the LocalPsf has a "native" representation of an image with
     *         predetermined dimensions.
     */
    virtual bool hasNativeImage() const { return false; }

    /**
     *  @brief Return the native dimensions of the image, or throw if !hasNativeImage().
     *
     *  The returned extent will always be odd in both x and y.
     */
    virtual geom::Extent2I getNativeImageDimensions() const {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicErrorException, "LocalPsf has no native image.h");
    }
    
    /**
     *  @brief Compute an image of the LocalPsf with the given dimensions, or throw if hasNativeImage().
     *
     *  The given extent must by odd in both x and y, and the image will be centered on the
     *  center of the middle pixel.
     *
     *  May or may not throw if hasNativeImage() is true.
     */
    virtual ndarray::Array<Pixel,2,2> computeImage(geom::Extent2I const & dimensions) const;

    /**
     *  @brief Return an image representation of the LocalPsf, or throw if !hasNativeImage().
     *
     *  The image will be centered on the center of the middle pixel, with dimensions given
     *  by getNativeImageDimensions().
     */
    virtual ndarray::Array<Pixel const,2,1> getNativeImage() const {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicErrorException, "LocalPsf has no native image.h");
    }

    /**
     *  @brief Return true if the LocalPsf has a "native" shapelet or multi-shapelet representation.
     */
    virtual bool hasNativeShapelet() const { return true; }

    /**
     *  @brief Return a shapelet representation of the LocalPsf with the given order and ellipse.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *  @param[in] order      Shapelet order.
     *  @param[in] ellipse    Ellipse to set the radius and ellipticity of the shapelet function.
     *
     *  May or may not throw if hasNativeShapelet() is true.
     */
    virtual Shapelet computeShapelet(
        math::shapelets::BasisTypeEnum basisType, 
        int order,
        geom::ellipses::BaseCore const & ellipse
    ) const {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicErrorException, "LocalPsf has no native image.h");
    }

    /**
     *  @brief Return a shapelet representation of the LocalPsf with the given order.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *  @param[in] order      Shapelet order.
     *
     *  Equivalent to computeShapelet(basisType, order, computeMoments());
     *  May or may not throw if hasNativeShapelet() is true.
     */
    Shapelet computeShapelet(math::shapelets::BasisTypeEnum basisType, int order) const {
        return computeShapelet(basisType, order, computeMoments());
    }

    /**
     *  @brief Return a native (multi)shapelet representation of the LocalPsf.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *
     *  If a Psf can provide enough information (i.e. more than a shapelet order and ellipse)
     *  to compute a multi-scale shapelet, it should have hasNativeShapelet()==true and
     *  implement this member function, even if it isn't natively stored in shapelet form.
     *
     *  Should throw if hasNativeShapelet() is false.
     */
    virtual MultiShapelet getNativeShapelet(math::shapelets::BasisTypeEnum basisType) const;

    /**
     *  @brief Compute the 2nd-order moments of the Psf.
     */
    virtual geom::ellipses::Quadrupole computeMoments() const;

    /**
     *  @brief Fill the pixels of the footprint with a point source model in the given flattened array.
     *
     *  @param[in]  fp     Footprint that defines the nonzero pixels of the model.
     *  @param[in]  point  Center of the point source in the same coordinate system as the footprint.
     *  @param[out] array  Flattened output array with size equal to footprint area.
     */
    virtual void evaluatePointSource(
        Footprint const & fp, 
        geom::Point2D const & point, 
        ndarray::Array<Pixel, 1, 0> const & array
    ) const;
    
    /**
     *  @brief Return a flattened array with a point source model corresponding to a footprint.
     *
     *  @param[in]  fp     Footprint that defines the nonzero pixels of the model.
     *  @param[in]  point  Center of the point source in the same coordinate system as the footprint.
     */
    ndarray::Array<Pixel, 1,1> evaluatePointSource(
        Footprint const & fp, 
        geom::Point2D const & point
    ) const {
        ndarray::Array<Pixel, 1, 1> array = ndarray::allocate(ndarray::makeVector(fp.getArea()));
        evaluatePointSource(fp, point, array);
        return array;
    }

private:
    MultiShapelet _shapelet;
};

}}}

#endif
