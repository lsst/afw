// -*- LSST-C++ -*-
#ifndef LSST_AFW_DETECTION_LOCAL_PSF_H
#define LSST_AFW_DETECTION_LOCAL_PSF_H

#include "boost/shared_ptr.hpp"
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

class LocalPsf {
public: 
    typedef boost::shared_ptr<LocalPsf> Ptr;
    typedef boost::shared_ptr<LocalPsf const> ConstPtr;

    typedef lsst::afw::math::shapelets::ShapeletFunction Shapelet;
    typedef lsst::afw::math::shapelets::MultiShapeletFunction MultiShapelet;

    typedef Psf::Pixel Pixel;
    typedef Psf::Image Image;

    /**
     *  @brief Return true if the LocalPsf has a "native" representation of an image with
     *         predetermined dimensions.
     */
    virtual bool hasNativeImage() const = 0;

    /**
     *  @brief Return the native dimensions of the image, or throw if !hasNativeImage().
     *
     *  The returned extent will always be odd in both x and y.
     */
    virtual geom::Extent2I getNativeImageDimensions() const = 0;
    
    /**
     *  @brief Compute an image of the LocalPsf with the given dimensions, or throw if hasNativeImage().
     *
     *  The given extent must by odd in both x and y, and the image will be centered on the
     *  center of the middle pixel.
     *
     *  May or may not throw if hasNativeImage() is true.
     */
    virtual ndarray::Array<Pixel,2,2> computeImage(geom::Extent2I const & dimensions) const = 0;

    /**
     *  @brief Return an image representation of the LocalPsf, or throw if !hasNativeImage().
     *
     *  The image will be centered on the center of the middle pixel, with dimensions given
     *  by getNativeImageDimensions().
     */
    virtual ndarray::Array<Pixel const,2,1> getNativeImage() const = 0;

    /**
     *  @brief Return true if the LocalPsf has a "native" shapelet or multi-shapelet representation.
     */
    virtual bool hasNativeShapelet() const = 0;

    /**
     *  @brief Return a shapelet representation of the LocalPsf with the given order and ellipse.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *  @param[in] order      Shapelet order.
     *  @param[in] ellipse    Ellipse to set the radius and ellipticity of the shapelet function.
     *
     *  Should throw if hasNativeShapelet() is true.
     */
    virtual Shapelet computeShapelet(
        math::shapelets::BasisTypeEnum basisType, 
        int order,
        geom::ellipses::BaseCore const & ellipse
    ) const = 0;

    /**
     *  @brief Return a shapelet representation of the LocalPsf with the given order.
     *
     *  @param[in] basisType  Shapelet basis to use (HERMITE or LAGUERRE).
     *  @param[in] order      Shapelet order.
     *
     *  Equivalent to computeShapelet(basisType, order, computeMoments());
     *  May throw if hasNativeShapelet() is true.
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
    virtual MultiShapelet getNativeShapelet(math::shapelets::BasisTypeEnum basisType) const = 0;

    /**
     *  @brief Compute the 2nd-order moments of the Psf.
     */
    virtual geom::ellipses::Quadrupole computeMoments() const = 0;

    /**
     *  @brief Fill the pixels of the footprint with a point source model in the given flattened array.
     *
     *  @param[in]  fp     Footprint to set the bounds of the model.
     *  @param[in]  point  Center of the point source in the same coordinate system as the footprint.
     *  @param[out] array  Flattened output array with size equal to footprint area.
     */
    virtual void evaluatePointSource(
        Footprint const & fp, 
        geom::Point2D const & point, 
        ndarray::Array<Pixel, 1, 0> const & array
    ) const = 0;
    
    /**
     *  @brief Return a flattened array with a point source model corresponding to a footprint.
     *
     *  @param[in]  fp     Footprint to set the bounds of the model.
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

    virtual ~LocalPsf() {}

};

}}}

#endif
