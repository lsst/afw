// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008 - 2012 LSST Corporation.
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

/**
 * @file
 *
 * @brief GPU accelerared image warping
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */
#ifndef LSST_AFW_MATH_DETAIL_SRCPOSFUNCTOR_H
#define LSST_AFW_MATH_DETAIL_SRCPOSFUNCTOR_H

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

    class SrcPosFunctor {
    public:
        typedef boost::shared_ptr<SrcPosFunctor> Ptr;

        explicit SrcPosFunctor() {};
        virtual ~SrcPosFunctor() {};

        virtual lsst::afw::geom::Point2D operator()(int destCol, int destRow) const = 0;
    };


    class WcsSrcPosFunctor : public SrcPosFunctor {
    public:
        typedef boost::shared_ptr<WcsSrcPosFunctor> Ptr;

        explicit WcsSrcPosFunctor(
            lsst::afw::geom::Point2D const &destXY0,    ///< xy0 of destination image
            lsst::afw::image::Wcs const &destWcs,       ///< WCS of remapped %image
            lsst::afw::image::Wcs const &srcWcs         ///< WCS of source %image
        ) :
            SrcPosFunctor(),
            _destXY0(destXY0),
            _destWcs(destWcs),
            _srcWcs(srcWcs)
        {}
        
        virtual ~WcsSrcPosFunctor() {};

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


    class AffineTransformSrcPosFunctor : public SrcPosFunctor {
    public:
        // NOTE: The transform will be called to locate a *source* pixel given a *dest* pixel
        // ... so we actually want to use the *inverse* transform of the affineTransform we we're given.
        // Thus _affineTransform is initialized to affineTransform.invert()
        AffineTransformSrcPosFunctor(
            lsst::afw::geom::Point2D const &destXY0,    ///< xy0 of destination image
            lsst::afw::geom::AffineTransform const &affineTransform
                ///< affine transformation mapping source position to destination position
        ) :
            SrcPosFunctor(),
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

#endif // !defined(LSST_AFW_MATH_DETAIL_SRCPOSFUNCTOR_H)
