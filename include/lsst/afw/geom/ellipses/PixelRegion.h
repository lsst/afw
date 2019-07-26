// -*- lsst-c++ -*-
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

#ifndef LSST_AFW_GEOM_ELLIPSES_PixelRegion_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_PixelRegion_h_INCLUDED

#include <vector>

#include "lsst/geom/Box.h"
#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/**
 *  A pixelized region containing all pixels whose centers are within an
 *  Ellipse.
 *
 *  The pixel region for an ellipse may be larger or smaller in area than
 *  the ellipse itself, depending on the details of where pixel centers land,
 *  and it may be empty even if the area of the ellipse is nonzero.
 */
class PixelRegion final {
public:

    /// Iterator type used by begin() and end().
    using Iterator = std::vector<Span>::const_iterator;

    /// Construct a PixelRegion from an Ellipse.
    explicit PixelRegion(Ellipse const& ellipse);

    //@{
    /// PixelRegion is copy and move constructable and assignable.
    PixelRegion(PixelRegion const&) = default;
    PixelRegion(PixelRegion&&) = default;
    PixelRegion& operator=(PixelRegion const&) = default;
    PixelRegion& operator=(PixelRegion&&) = default;
    ~PixelRegion() = default;
    //@}

    //@{
    /**
     *  Iterator range over Spans whose pixels are within the Ellipse.
     */
    Iterator begin() const { return _spans.begin(); }
    Iterator end() const { return _spans.end(); }
    //@}

    /**
     * Return the bounding box of the pixel region.
     *
     * This is guaranteed to be the smallest box that includes all Spans.  It
     * has no guaranteed relationship with the Ellipse's direct
     * (floating-point) bounding box as computed by Ellipse::computeBBox(), and
     * may be empty even if the Ellipse's bounding box is not.
     */
    lsst::geom::Box2I const& getBBox() const { return _bbox; }

    /**
     * Return the span at the given y coordinate value.
     *
     * @throws lsst::pex::exceptions::OutOfRangeError Thrown if y is not
     *      within the y bounds of getBBox().
     */
    Span const getSpanAt(int y) const;

private:
    std::vector<Span> _spans;
    lsst::geom::Box2I _bbox;
};

}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_GEOM_ELLIPSES_PixelRegion_h_INCLUDED
