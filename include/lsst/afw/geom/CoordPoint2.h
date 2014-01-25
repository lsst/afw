/* 
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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
 
#if !defined(LSST_AFW_GEOM_COORDPOINT2_H)
#define LSST_AFW_GEOM_COORDPOINT2_H

#include <string>
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/CoordSys.h"

namespace lsst {
namespace afw {
namespace geom {

/**
 * A 2d point with associated coordinate system
 */
class CoordPoint2 {
public:
    /**
     * Construct a CoordPoint2
     */
    explicit CoordPoint2(
        Point2D const &point, ///< 2D point
        CoordSys const &coordSys    ///< coordinate system
    ) : _point(point), _coordSys(coordSys) {}

    Point2D getPoint() const { return _point; }

    CoordSys getCoordSys() const { return _coordSys; }

private:
    Point2D _point;   ///< 2D point
    CoordSys _coordSys;     ///< coordinate system
};

}}}

#endif
