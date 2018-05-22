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

#if !defined(LSST_AFW_IMAGE_DEFECT_H)
#define LSST_AFW_IMAGE_DEFECT_H
/**
 * A base class for image defects
 */
#include <limits>
#include <vector>
#include "lsst/geom.h"

namespace lsst {
namespace afw {
namespace image {

/**
 * Encapsulate information about a bad portion of a detector
 */
class DefectBase {
public:
    explicit DefectBase(const lsst::geom::Box2I& bbox  ///< Bad pixels' bounding box
                        )
            : _bbox(bbox) {}
    DefectBase(DefectBase const&) = default;
    DefectBase(DefectBase&&) = default;
    DefectBase& operator=(DefectBase const&) = default;
    DefectBase& operator=(DefectBase&&) = default;
    virtual ~DefectBase() = default;

    lsst::geom::Box2I const& getBBox() const { return _bbox; }  ///< Return the Defect's bounding box
    int getX0() const { return _bbox.getMinX(); }         ///< Return the Defect's left column
    int getX1() const { return _bbox.getMaxX(); }         ///< Return the Defect's right column
    int getY0() const { return _bbox.getMinY(); }         ///< Return the Defect's bottom row
    int getY1() const { return _bbox.getMaxY(); }         ///< Return the Defect's top row

    void clip(lsst::geom::Box2I const& bbox) { _bbox.clip(bbox); }

    /**
     * Offset a Defect by `(dx, dy)`
     */
    void shift(int dx,  ///< How much to move defect in column direction
               int dy   ///< How much to move in row direction
               ) {
        _bbox.shift(lsst::geom::Extent2I(dx, dy));
    }
    void shift(lsst::geom::Extent2I const& d) { _bbox.shift(d); }

private:
    lsst::geom::Box2I _bbox;  ///< Bounding box for bad pixels
};
}
}
}

#endif
