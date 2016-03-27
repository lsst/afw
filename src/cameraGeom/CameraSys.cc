/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
