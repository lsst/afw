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
#include "boost/make_shared.hpp"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/cameraGeom/CameraSys.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * A 2d point in some camera coordinate system
 */
class CameraPoint {
public:
    /**
     * Construct a CameraPoint
     */
    explicit CameraPoint(
        geom::Point2D const &point,     ///< 2D point
        CONST_PTR(CameraSys) const &cameraSys   ///< camera system
    ) : _point(point), _cameraSys(cameraSys) {}

    explicit CameraPoint(
        geom::Point2D const &point,     ///< 2D point
        std::string const &coordSys,    ///< coordinate system
        std::string const &frameName = ""  ///< frame name, if required to disambiguate, else ""
    ) : _point(point), _cameraSys(boost::make_shared<CameraSys>(coordSys, frameName)) {}

    geom::Point2D getPoint() const { return _point; }

    CONST_PTR(CameraSys) getCameraSys() const { return _cameraSys; }

    std::string getCoordSys() const { return _cameraSys->getCoordSys(); }

    std::string getFrameName() const { return _cameraSys->getFrameName(); }
 
private:
    geom::Point2D _point;   ///< 2D point
    CONST_PTR(CameraSys) _cameraSys;  ///< camera system
};

}}}

#endif
