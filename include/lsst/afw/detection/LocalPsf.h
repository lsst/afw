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
#ifndef LSST_AFW_DETECTION_LOCALPSF_H
#define LSST_AFW_DETECTION_LOCALPSF_H

#include "lsst/base.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/shapelets.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/ndarray.h"

namespace lsst {
namespace afw {
namespace detection {

/**
 *  @brief A spatially-invariant representation of a Psf evaluated at a point and color.
 *
 *  LocalPsf is intended to be ultimately be useful as a place to put all local evaluations of
 *  a Psf, including images.  Right now it's just a way to get shapelets out of an arbitrary
 *  Psf - only meas/multifit uses LocalPsf right now, and that only needs shapelets and
 *  evaluatePointSource().
 */
class LocalPsf {
public: 
    typedef boost::shared_ptr<LocalPsf> Ptr;
    typedef boost::shared_ptr<LocalPsf const> ConstPtr;

    typedef lsst::afw::math::shapelets::ShapeletFunction Shapelet;
    typedef lsst::afw::math::shapelets::MultiShapeletFunction MultiShapelet;

    typedef Psf::Pixel Pixel;
    typedef Psf::Image Image;

    /**
     *  @brief Return the point the LocalPsf was evaluated at.
     */
    lsst::afw::geom::Point2D const & getPoint() const { return _point; }

    /**
     *  @brief Return true if the LocalPsf has a "native" shapelet or multi-shapelet representation.
     */
    virtual bool hasNativeShapelet() const { return false; }

    /**
     *  @brief Return a shapelet representation of the LocalPsf with the given order and ellipse.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *  @param[in] order      Shapelet order.
     *  @param[in] ellipse    Ellipse to set the radius, ellipticity, and center of the shapelet function.
     *
     *  The shapelet representation is guaranteed to be normalized such that it integrates to one.
     *  May throw if hasNativeShapelet() is true.
     */
    virtual Shapelet computeShapelet(
        lsst::afw::math::shapelets::BasisTypeEnum basisType, 
        int order,
        lsst::afw::geom::ellipses::Ellipse const & ellipse
    ) const = 0;

    /**
     *  @brief Return a shapelet representation of the LocalPsf with the given order.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *  @param[in] order      Shapelet order.
     *
     *  Equivalent to computeShapelet(basisType, order, computeMoments());
     *  The shapelet representation is guaranteed to be normalized such that it integrates to one.
     *  May throw if hasNativeShapelet() is true.
     */
    Shapelet computeShapelet(lsst::afw::math::shapelets::BasisTypeEnum basisType, int order) const {
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
     *  The shapelet representation is guaranteed to be normalized such that it integrates to one.
     *  Should throw if hasNativeShapelet() is false.
     */
    virtual MultiShapelet getNativeShapelet(lsst::afw::math::shapelets::BasisTypeEnum basisType) const {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicErrorException,
                          "LocalPsf has no native shapelet representation.");
    }

    /**
     *  @brief Compute the 1st and 2nd-order moments of the Psf.
     */
    virtual lsst::afw::geom::ellipses::Ellipse computeMoments() const = 0;

    /**
     *  @brief Fill the pixels of the footprint with a point source model in the given flattened array.
     *
     *  @param[in]  fp     Footprint that defines the nonzero pixels of the model.
     *  @param[out] array  Flattened output array with size equal to footprint area.
     *  @param[in]  offset Offset of the point source from the point the LocalPsf was evaluated at.
     */
    virtual void evaluatePointSource(
        Footprint const & fp, 
        lsst::ndarray::Array<Pixel,1,0> const & array,
        lsst::afw::geom::Extent2D const & offset = lsst::afw::geom::Extent2D()
    ) const = 0;
    
    /**
     *  @brief Return a flattened array with a point source model corresponding to a footprint.
     *
     *  @param[in]  fp     Footprint that defines the nonzero pixels of the model.
     *  @param[in]  offset Offset of the point source from the point the LocalPsf was evaluated at.
     */
    lsst::ndarray::Array<Pixel,1,1> evaluatePointSource(
        Footprint const & fp, 
        lsst::afw::geom::Extent2D const & offset = lsst::afw::geom::Extent2D()
    ) const {
        ndarray::Array<Pixel,1,1> array = ndarray::allocate(ndarray::makeVector(fp.getArea()));
        evaluatePointSource(fp, array, offset);
        return array;
    }

    virtual ~LocalPsf() {}

protected:
    explicit LocalPsf(geom::Point2D const & point) : _point(point) {}

private:
    void operator=(LocalPsf const &); // disabled
    geom::Point2D _point;
};


/**
 *  @brief A LocalPsf for multi-shapelet and multi-Gaussian PSFs.
 */
class ShapeletLocalPsf : public LocalPsf {
public: 
    typedef boost::shared_ptr<ShapeletLocalPsf> Ptr;
    typedef boost::shared_ptr<ShapeletLocalPsf const> ConstPtr;

    /**
     *  @brief Return true if the LocalPsf has a "native" shapelet or multi-shapelet representation.
     */
    virtual bool hasNativeShapelet() const { return true; }

    /**
     *  @brief Return a shapelet representation of the LocalPsf with the given order and ellipse.
     *
     *  Not implemented by ShapeletLocalPsf; use getNativeShapelet().
     */
    virtual Shapelet computeShapelet(
        lsst::afw::math::shapelets::BasisTypeEnum basisType, 
        int order,
        lsst::afw::geom::ellipses::Ellipse const & ellipse
    ) const {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicErrorException,
                          "LocalPsf does not support computeShapelet(); use getNativeShapelet().");        
    }

    /**
     *  @brief Return a native (multi)shapelet representation of the LocalPsf.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *
     *  The shapelet representation is guaranteed to be normalized such that it integrates to one.
     */
    virtual MultiShapelet getNativeShapelet(lsst::afw::math::shapelets::BasisTypeEnum basisType) const {
        return _shapelet;
    }

    /**
     *  @brief Compute the 1st and 2nd-order moments of the Psf.
     */
    virtual lsst::afw::geom::ellipses::Ellipse computeMoments() const {
        return _shapelet.evaluate().computeMoments();
    }

    /**
     *  @brief Fill the pixels of the footprint with a point source model in the given flattened array.
     *
     *  @param[in]  fp     Footprint that defines the nonzero pixels of the model.
     *  @param[out] array  Flattened output array with size equal to footprint area.
     *  @param[in]  offset Offset of the point source from the point the LocalPsf was evaluated at.
     */
    virtual void evaluatePointSource(
        Footprint const & fp, 
        lsst::ndarray::Array<Pixel,1,0> const & array,
        lsst::afw::geom::Extent2D const & offset = lsst::afw::geom::Extent2D()
    ) const;

    explicit ShapeletLocalPsf(lsst::afw::geom::Point2D const & center, MultiShapelet const & shapelet) :
        LocalPsf(center), _shapelet(shapelet)
    {
        _shapelet.normalize();
    }

private:
    MultiShapelet _shapelet;
};

/**
 *  @brief An image-based LocalPsf.
 */
class ImageLocalPsf : public LocalPsf {
public: 
    typedef boost::shared_ptr<ImageLocalPsf> Ptr;
    typedef boost::shared_ptr<ImageLocalPsf const> ConstPtr;

    /**
     *  @brief Return a shapelet representation of the LocalPsf with the given order and ellipse.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *  @param[in] order      Shapelet order.
     *  @param[in] ellipse    Ellipse to set the radius, ellipticity, and center of the shapelet function.
     *
     *  The shapelet representation is guaranteed to be normalized such that it integrates to one.
     */
    virtual Shapelet computeShapelet(
        lsst::afw::math::shapelets::BasisTypeEnum basisType, 
        int order,
        lsst::afw::geom::ellipses::Ellipse const & ellipse
    ) const;

    /**
     *  @brief Compute the 1st and 2nd-order moments of the Psf.
     */
    virtual lsst::afw::geom::ellipses::Ellipse computeMoments() const;

    /**
     *  @brief Fill the pixels of the footprint with a point source model in the given flattened array.
     *
     *  @param[in]  fp     Footprint that defines the nonzero pixels of the model.
     *  @param[out] array  Flattened output array with size equal to footprint area.
     *  @param[in]  offset Offset of the point source from the point the LocalPsf was evaluated at.
     */
    virtual void evaluatePointSource(
        Footprint const & fp, 
        lsst::ndarray::Array<Pixel,1,0> const & array,
        lsst::afw::geom::Extent2D const & offset = lsst::afw::geom::Extent2D()
    ) const;

    /**
     *  @brief Construct a new ImageLocalPsf with the given center and image.
     *
     *  The image will be shallow-copied and must already be normalized to sum to one, and it
     *  must be centered at the given center point on the image with xy0 taken into account
     *  (i.e., like the output of Psf::computeImage, but normalized).
     */
    explicit ImageLocalPsf(lsst::afw::geom::Point2D const & center, Image const & image) :
        LocalPsf(center), _image(image)
    {}

protected:
    Image _image;
};

}}}

#endif
