/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "lsst/afw/cameraGeom/Orientation.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

    Orientation::Orientation(
        geom::Point2D const fpPosition,
        geom::Point2D const refPoint,
        geom::Angle const yaw,
        geom::Angle const pitch,
        geom::Angle const roll
    ) :
        _fpPosition(fpPosition),
        _refPoint(refPoint),
        _yaw(yaw),
        _pitch(pitch),
        _roll(roll),
        _rotMat()
    {
        double cosYaw = std::cos(_yaw);
        double sinYaw = std::sin(_yaw);
        double cosPitch = std::cos(_pitch);
        double sinPitch = std::sin(_pitch);
        double cosRoll = std::cos(_roll);
        double sinRoll = std::sin(_roll);

        // This comes from the rotation matrix written down here:
        // http://en.wikipedia.org/wiki/Euler_angles
        // for Tait-Bryan angles Z_1Y_2X_3
        // _rotMat = coeffA  coeffB
        //           coeffD  coeffE
        _rotMat << cosYaw*cosPitch, cosYaw*sinPitch*sinRoll - cosRoll*sinYaw,
                   cosPitch*sinYaw,  cosYaw*cosRoll + sinYaw*sinPitch*sinRoll;

    }

    /// Return the number of quarter turns (rounded to the closest quarter)
    int Orientation::getNQuarter() const {
        float yawDeg = _yaw.asDegrees();
        while (yawDeg < 0.) {
            yawDeg += 360.;
        }
        return std::floor((yawDeg + 45.)/90.);
    }

    geom::AffineXYTransform Orientation::makePixelFpTransform(
            geom::Extent2D const pixelSizeMm
    ) const {
        // jacobian = coeffA*pixelSizeMmX, coeffB*pixelSizeMmY,
        //            coeffD*pixelSizeMmX, coeffE*pixelSizeMmY
        Eigen::Matrix2d jacobian = _rotMat.array() \
            * (Eigen::Vector2d::Ones() * pixelSizeMm.asEigen().transpose()).array();

        Eigen::Vector2d refMm = pixelSizeMm.asEigen().array() * _refPoint.asEigen().array();
        Eigen::Vector2d translation = _fpPosition.asEigen() - (_rotMat * refMm);

        geom::AffineTransform affineTransform = geom::AffineTransform(jacobian, translation);
        return geom::AffineXYTransform(affineTransform);
    }

    geom::AffineXYTransform Orientation::makeFpPixelTransform(
            geom::Extent2D const pixelSizeMm
    ) const {
        return geom::AffineXYTransform(makePixelFpTransform(pixelSizeMm).getReverseTransform());
    }

}}}
