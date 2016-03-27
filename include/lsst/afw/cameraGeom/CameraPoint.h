/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
