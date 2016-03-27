/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "lsst/afw/cameraGeom/CameraPoint.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

std::ostream &operator<< (std::ostream &os, CameraPoint const &cameraPoint) {
    os << "CameraPoint(" << cameraPoint.getPoint() << ", " << cameraPoint.getCameraSys() << ")";
    return os;
}

}}}
