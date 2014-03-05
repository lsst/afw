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

#include "lsst/afw/cameraGeom/CameraSys.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

CameraSys const FOCAL_PLANE = CameraSys("FocalPlane");

CameraSys const PUPIL = CameraSys("Pupil");

CameraSysPrefix const PIXELS = CameraSysPrefix("Pixels");

CameraSysPrefix const TAN_PIXELS = CameraSysPrefix("TanPixels");

CameraSysPrefix const ACTUAL_PIXELS = CameraSysPrefix("ActualPixels");

std::ostream &operator<< (std::ostream &os, CameraSysPrefix const &camSysPrefix) {
    os << "CameraSysPrefix(" << camSysPrefix.getSysName() << ")";
    return os;
}

std::ostream &operator<< (std::ostream &os, CameraSys const &cameraSys) {
    os << "CameraSys(" << cameraSys.getSysName();
    if (cameraSys.hasDetectorName()) {
        os << ", " << cameraSys.getDetectorName();
    }
    os << ")";
    return os;
}
}
// instantiate CameraTransformMap = TransformMap<CameraSys>
template class geom::TransformMap<cameraGeom::CameraSys>;
}}
