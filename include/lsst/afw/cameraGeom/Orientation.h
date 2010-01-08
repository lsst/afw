#if !defined(LSST_AFW_CAMERAGEOM_ORIENTATION_H)
#define LSST_AFW_CAMERAGEOM_ORIENTATION_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"

/**
 * @file
 *
 * Describe a Detector's orientation
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

namespace afwGeom = lsst::afw::geom;

/**
 * Describe a detector's orientation
 */
class Orientation {
public:
    explicit Orientation(double pitch=0.0, ///< pitch (rotation in YZ), radians
                         double roll=0.0,  ///< roll (rotation in XZ), radians
                         double yaw=0.0) : ///< yaw (rotation in XY), radians
        _pitch(pitch), _roll(roll), _yaw(yaw) {}
    /// Return the pitch angle
    double getPitch() const { return _pitch; }
    /// Return the roll angle
    double getRoll() const { return _roll; }
    /// Return the yaw angle
    double getYaw() const { return _yaw; }
private:
    double _pitch;                         // pitch
    double _roll;                          // roll
    double _yaw;                           // yaw
};

}}}

#endif
