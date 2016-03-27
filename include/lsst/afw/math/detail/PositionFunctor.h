// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief GPU accelerared image warping
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */
#ifndef LSST_AFW_MATH_DETAIL_POSITIONFUNCTOR_H
#define LSST_AFW_MATH_DETAIL_POSITIONFUNCTOR_H

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

    /**
     * @brief Base class to transform pixel position for a destination image
     *        to its position in the original source image.
     *
     * The different possible transform definitions (from WCS to WCS, or via AffineTransform) are handled
     * through derived classes, and are used in warping.  When computing a warped image, one
     * iterates over the pixels in the destination image and must ask 'for the value I wish to
     * put *here*, where should I go to find it in the source image?'.  Instantiating a Functor derived from
     * this base class creates a callable function which accepts (destination) col,row and returns
     * (source image) col,row (in the form of a Point2D).
     */    
    class PositionFunctor {
    public:
        typedef boost::shared_ptr<PositionFunctor> Ptr;

        explicit PositionFunctor() {};
        virtual ~PositionFunctor() {};

        virtual lsst::afw::geom::Point2D operator()(int destCol, int destRow) const = 0;
    };


    /**
     * @brief Derived functor class to transform pixel position for a destination image
     *        to its position in the source image.  The transform is from one WCS to another.
     */    
    class WcsPositionFunctor : public PositionFunctor {
    public:
        typedef boost::shared_ptr<WcsPositionFunctor> Ptr;

        explicit WcsPositionFunctor(
            lsst::afw::geom::Point2D const &destXY0,    ///< xy0 of destination image
            lsst::afw::image::Wcs const &destWcs,       ///< WCS of remapped %image
            lsst::afw::image::Wcs const &srcWcs         ///< WCS of source %image
        ) :
            PositionFunctor(),
            _destXY0(destXY0),
            _destWcs(destWcs),
            _srcWcs(srcWcs)
        {}
        
        virtual ~WcsPositionFunctor() {};

        virtual lsst::afw::geom::Point2D operator()(int destCol, int destRow) const {
            double const col = lsst::afw::image::indexToPosition(destCol + _destXY0[0]);
            double const row = lsst::afw::image::indexToPosition(destRow + _destXY0[1]);
            lsst::afw::geom::Angle sky1, sky2;
            _destWcs.pixelToSky(col, row, sky1, sky2);
            return _srcWcs.skyToPixel(sky1, sky2);
        }

    private:
        lsst::afw::geom::Point2D const &_destXY0;
        lsst::afw::image::Wcs const &_destWcs;
        lsst::afw::image::Wcs const &_srcWcs;
    };


    /**
     * @brief Derived functor class to transform pixel position for a destination image
     *        to its position in the source image via an AffineTransform.
     */    
    class AffineTransformPositionFunctor : public PositionFunctor {
    public:
        // NOTE: The transform will be called to locate a *source* pixel given a *dest* pixel
        // ... so we actually want to use the *inverse* transform of the affineTransform we were given.
        // Thus _affineTransform is initialized to affineTransform.invert()
        AffineTransformPositionFunctor(
            lsst::afw::geom::Point2D const &destXY0,    ///< xy0 of destination image
            lsst::afw::geom::AffineTransform const &affineTransform
                ///< affine transformation mapping source position to destination position
        ) :
            PositionFunctor(),
            _destXY0(destXY0),
            _affineTransform() {
            _affineTransform = affineTransform.invert();
        }

        virtual lsst::afw::geom::Point2D operator()(int destCol, int destRow) const {
            double const col = lsst::afw::image::indexToPosition(destCol + _destXY0[0]);
            double const row = lsst::afw::image::indexToPosition(destRow + _destXY0[1]);
            lsst::afw::geom::Point2D p = _affineTransform(lsst::afw::geom::Point2D(col, row));
            return p;
        }
    private:
        lsst::afw::geom::Point2D const &_destXY0;
        lsst::afw::geom::AffineTransform _affineTransform;
    };

}}}} // namespace lsst::afw::math::detail

#endif // !defined(LSST_AFW_MATH_DETAIL_POSITIONFUNCTOR_H)
