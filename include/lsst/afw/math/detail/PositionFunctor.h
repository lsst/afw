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

/*
 * image warping
 */
#ifndef LSST_AFW_MATH_DETAIL_POSITIONFUNCTOR_H
#define LSST_AFW_MATH_DETAIL_POSITIONFUNCTOR_H

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/XYTransform.h"
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
    explicit PositionFunctor(){};
    virtual ~PositionFunctor(){};

    virtual lsst::afw::geom::Point2D operator()(int destCol, int destRow) const = 0;
};

/**
 * Functor class that wraps an XYTransform
 */
class XYTransformPositionFunctor : public PositionFunctor {
public:
    explicit XYTransformPositionFunctor(
            lsst::afw::geom::Point2D const &destXY0,         ///< xy0 of destination image
            lsst::afw::geom::XYTransform const &XYTransform  ///< xy transform mapping source position
            ///< to destination position in the forward direction (but only the reverse direction is used)
            )
            : PositionFunctor(), _destXY0(destXY0), _xyTransformPtr(XYTransform.clone()) {}

    virtual ~XYTransformPositionFunctor(){};

    virtual lsst::afw::geom::Point2D operator()(int destCol, int destRow) const {
        afw::geom::Point2D const destPos{lsst::afw::image::indexToPosition(destCol + _destXY0[0]),
                                         lsst::afw::image::indexToPosition(destRow + _destXY0[1])};
        return _xyTransformPtr->reverseTransform(destPos);
    }

private:
    lsst::afw::geom::Point2D const _destXY0;
    std::shared_ptr<lsst::afw::geom::XYTransform const> _xyTransformPtr;
};
}
}
}
}  // namespace lsst::afw::math::detail

#endif  // !defined(LSST_AFW_MATH_DETAIL_POSITIONFUNCTOR_H)
