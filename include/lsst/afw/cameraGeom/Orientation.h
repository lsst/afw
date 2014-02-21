/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 
#if !defined(LSST_AFW_CAMERAGEOM_ORIENTATION_H)
#define LSST_AFW_CAMERAGEOM_ORIENTATION_H

#include <string>
#include <cmath>
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/image/Utils.h"

/**
 * @file
 *
 * Describe a Detector's orientation
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Describe a detector's orientation in the focal plane
 *
 * All rotations are about the reference point on the detector.
 * Rotations are intrinsic, meaning each rotation is applied in the coordinates system
 * produced by the previous rotation.
 * Rotations are applied in this order: yaw (Z), pitch (Y'), and roll (X'').
 *
 * @warning: default refPoint is -0.5, -0.5 (the lower left corner of a detector).
 * This means that the default-constructed Orientation is not a unity transform,
 * but instead includes a 1/2 pixel shift.
 */
class Orientation {
public:
    explicit Orientation(
        geom::Point2D const fpPosition=geom::Point2D(0, 0),
            ///< Focal plane position of detector reference point (mm)
        geom::Point2D const refPoint=geom::Point2D(-0.5, -0.5),
            ///< Reference point on detector (pixels).
            ///< Offset is measured to this point and all all rotations are about this point.
            ///< The default value (-0.5, -0.5) is the lower left corner of the detector.
        geom::Angle const yaw=geom::Angle(0),   ///< yaw: rotation about Z (X to Y), 1st rotation
        geom::Angle const pitch=geom::Angle(0),  ///< pitch: rotation about Y' (Z'=Z to X'), 2nd rotation
        geom::Angle const roll=geom::Angle(0)  ///< roll: rotation about X'' (Y''=Y' to Z''), 3rd rotation
    ) :
        _fpPosition(fpPosition), _refPoint(refPoint),
        _yaw(yaw), _cosYaw(std::cos(yaw)),  _sinYaw(std::sin(yaw)),
        _pitch(pitch), _cosPitch(std::cos(pitch)),  _sinPitch(std::sin(pitch)),
        _roll(roll), _cosRoll(std::cos(roll)),  _sinRoll(std::sin(roll))
    {
        // This comes from the rotation matrix written down here:
        // http://en.wikipedia.org/wiki/Euler_angles
        // for Tait-Bryan angles Z_1Y_2X_3
        _coeffA = _cosYaw*_cosPitch;
        _coeffB = _cosYaw*_sinPitch*_sinRoll - _cosRoll*_sinYaw;
        _coeffD = _cosPitch*_sinYaw;
        _coeffE = _cosYaw*_cosRoll + _sinYaw*_sinPitch*_sinRoll;
    }

    /// Return focal plane position of detector reference point (mm)
    geom::Point2D getFpPosition() const { return _fpPosition; }

    /// Return detector reference point (pixels)
    geom::Point2D getReferencePoint() const { return _refPoint; }

    /// Return the pitch angle
    lsst::afw::geom::Angle getPitch() const { return _pitch; }

    /// Return the roll angle
    geom::Angle getRoll() const { return _roll; }

    /// Return the yaw angle
    geom::Angle getYaw() const { return _yaw; }

    /**
     * @brief Generate an XYTransform from pixel to focal plane coordinates
     *
     * @return lsst::afw::geom::AffineXYTransform from pixel to focal plane coordinates
     */
    geom::AffineXYTransform makePixelFpTransform(
            geom::Extent2D const pixelSizeMm ///< Size of the pixel in mm in X and Y
    ) const {
        Eigen::Matrix2d jacobian;
        jacobian << _coeffA*pixelSizeMm.getX(), _coeffB*pixelSizeMm.getY(),
                    _coeffD*pixelSizeMm.getX(), _coeffE*pixelSizeMm.getY();

        Eigen::Vector2d translation; 
        translation << _fpPosition.getX() - pixelSizeMm.getX()*_refPoint.getX(), 
                       _fpPosition.getY() - pixelSizeMm.getY()*_refPoint.getY();

        geom::AffineTransform affineTransform = geom::AffineTransform(jacobian, translation);
        return geom::AffineXYTransform(affineTransform);
    }

    /**
     * @brief Generate an XYTransform from focal plane to pixel coordinates
     *
     * @return lsst::afw::geom::AffineXYTransform from focal plane to pixel coordinates
     */
    geom::InvertedXYTransform makeFpPixelTransform(
            geom::Extent2D const pixelSizeMm ///< Size of the pixel in mm in X and Y
    ) const {
        geom::AffineXYTransform transform = makePixelFpTransform(pixelSizeMm);
  
        return geom::InvertedXYTransform(transform.clone());
    }
private:
    geom::Point2D _fpPosition;          ///< focal plane position of reference point on detector
    geom::Point2D _refPoint;            ///< reference point on detector

    lsst::afw::geom::Angle _yaw;        ///< yaw
    double _cosYaw;                     ///< cos(yaw)
    double _sinYaw;                     ///< sin(yaw)

    lsst::afw::geom::Angle _pitch;      ///< pitch
    double _cosPitch;                   ///< cos(pitch)
    double _sinPitch;                   ///< sin(pitch)

    lsst::afw::geom::Angle _roll;       ///< roll
    double _cosRoll;                    ///< cos(roll)
    double _sinRoll;                    ///< sin(roll)

    // Elements of the Jacobian for three space rotation projected into XY plane.
    double _coeffA;                     ///< 01 element
    double _coeffB;                     ///< 11 element
    double _coeffD;                     ///< 00 element
    double _coeffE;                     ///< 10 element
};

}}}

#endif
