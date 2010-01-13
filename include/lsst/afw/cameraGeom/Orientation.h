#if !defined(LSST_AFW_CAMERAGEOM_ORIENTATION_H)
#define LSST_AFW_CAMERAGEOM_ORIENTATION_H

#include <string>
#include <cmath>
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
 * Describe a detector's orientation with respect to the nominal position
 */
class Orientation {
public:
    explicit Orientation(int nQuarter = 0, ///< Nominal rotation of device in units of pi/2
                         double pitch=0.0, ///< pitch (rotation in YZ), radians
                         double roll=0.0,  ///< roll (rotation in XZ), radians
                         double yaw=0.0) : ///< yaw (rotation in XY), radians
        _nQuarter(nQuarter % 4),
        _pitch(pitch), _cosPitch(std::cos(pitch)),  _sinPitch(std::sin(pitch)),
        _roll(roll), _cosRoll(std::cos(roll)),  _sinRoll(std::sin(roll)),
        _yaw(yaw), _cosYaw(std::cos(yaw)),  _sinYaw(std::sin(yaw))
        {}

    /// Return the number of quarter-turns applied to this detector
    int getNQuarter() const { return _nQuarter; }

    /// Return the pitch angle
    double getPitch() const { return _pitch; }
    /// Return cos(pitch)
    double getCosPitch() const { return _cosPitch; }
    /// Return sin(pitch)
    double getSinPitch() const { return _sinPitch; }

    /// Return the roll angle
    double getRoll() const { return _roll; }
    /// Return cos(roll)
    double getCosRoll() const { return _cosRoll; }
    /// Return sin(roll)
    double getSinRoll() const { return _sinRoll; }

    /// Return the yaw angle
    double getYaw() const { return _yaw; }
    /// Return cos(yaw)
    double getCosYaw() const { return _cosYaw; }
    /// Return sin(yaw)
    double getSinYaw() const { return _sinYaw; }
private:
    int _nQuarter;                      // number of quarter-turns in +ve direction

    double _pitch;                      // pitch
    double _cosPitch;                   // cos(pitch)
    double _sinPitch;                   // sin(pitch)

    double _roll;                       // roll
    double _cosRoll;                    // cos(roll)
    double _sinRoll;                    // sin(roll)

    double _yaw;                        // yaw
    double _cosYaw;                     // cos(yaw)
    double _sinYaw;                     // sin(yaw)
};

}}}

#endif
