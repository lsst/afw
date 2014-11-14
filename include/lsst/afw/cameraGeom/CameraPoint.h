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
 
#if !defined(LSST_AFW_CAMERAGEOM_CAMERAPOINT_H)
#define LSST_AFW_CAMERAGEOM_CAMERAPOINT_H

#include <string>
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/cameraGeom/CameraSys.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * A Point2D with associated camera coordinate system
 */
class CameraPoint {
public:
    CameraPoint(geom::Point2D point, CameraSys const &cameraSys) : _point(point), _cameraSys(cameraSys) {}
    geom::Point2D getPoint() const { return _point; }
    CameraSys getCameraSys() const { return _cameraSys; }

    bool operator==(CameraPoint const &other) const {
        return (this->getPoint() == other.getPoint()) && (this->getCameraSys() == other.getCameraSys()); }

    bool operator!=(CameraPoint const &other) const { return !(*this == other); }

private:
    geom::Point2D _point;         ///< 2-d point
    CameraSys _cameraSys;   ///< camera coordinate system
};

std::ostream &operator<< (std::ostream &os, CameraPoint const &cameraPoint);

}}}

#endif
