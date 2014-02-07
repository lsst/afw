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
 * Describe a detector's orientation with respect to the nominal position
 *
 * All rotations are about the center of the detector coordinate system.
 * All rotations are intrinsic, so about the rotated coordinate system.
 * It this implementation applies the rotations in zy'x'' order.
 */
class Orientation {
public:
    explicit Orientation(geom::Point2D const offset=geom::Point2D(0, 0), ///< offset to the center of the detector (mm)
                         geom::Angle const yaw=geom::Angle(0),    ///< yaw (rotation in XY)
                         geom::Angle const roll=geom::Angle(0), ///< pitch (rotation in YZ)
                         geom::Angle const pitch=geom::Angle(0)  ///< roll (rotation in XZ)
                        )
        :
        _offset(offset),
        _yaw(yaw), _cosYaw(std::cos(yaw)),  _sinYaw(std::sin(yaw)),
        _roll(roll), _cosRoll(std::cos(roll)),  _sinRoll(std::sin(roll)),
        _pitch(pitch), _cosPitch(std::cos(pitch)),  _sinPitch(std::sin(pitch))
        {
            //This comes from the rotation matrix written down here:
            //http://en.wikipedia.org/wiki/Euler_angles
            //for Tait-Bryan angles Z_1Y_2X_3
            _coeffA = _cosYaw*_cosPitch;
            _coeffB = _cosYaw*_sinPitch*_sinRoll - _cosRoll*_sinYaw;
            _coeffD = _cosPitch*_sinYaw;
            _coeffE = _cosYaw*_cosRoll + _sinYaw*_sinPitch*_sinRoll;
        }
    /// Return offset
    lsst::afw::geom::Point2D getOffset() const { return _offset; }

    /// Return the pitch angle
    lsst::afw::geom::Angle getPitch() const { return _pitch; }
    /// Return cos(pitch)
    double getCosPitch() const { return _cosPitch; }
    /// Return sin(pitch)
    double getSinPitch() const { return _sinPitch; }

    /// Return the roll angle
    lsst::afw::geom::Angle getRoll() const { return _roll; }
    /// Return cos(roll)
    double getCosRoll() const { return _cosRoll; }
    /// Return sin(roll)
    double getSinRoll() const { return _sinRoll; }

    /// Return the yaw angle
    lsst::afw::geom::Angle getYaw() const { return _yaw; }
    /// Return cos(yaw)
    double getCosYaw() const { return _cosYaw; }
    /// Return sin(yaw)
    double getSinYaw() const { return _sinYaw; }

    lsst::afw::geom::AffineXYTransform makeFpPixelTransform(lsst::afw::geom::Extent2D const pixelSizeMm, lsst::afw::geom::Extent2I const numPixels) {
        //This comes from solving the transform equations.
        Eigen::Matrix2d jacobian;
        Eigen::Vector2d translation;
        lsst::afw::geom::AffineTransform affineTransform;
        double aprime = _coeffA*pixelSizeMm.getX();
        double bprime = _coeffB*pixelSizeMm.getY();
        double eprime = _coeffE*pixelSizeMm.getX();
        double dprime = _coeffD*pixelSizeMm.getY();
        double numer = dprime*bprime/aprime - eprime;
        double xoffprime = _offset.getX() - pixelSizeMm.getX()*numPixels.getX()/2.;
        double yoffprime = _offset.getY() - pixelSizeMm.getY()*numPixels.getY()/2.;
        jacobian(0,0) = 1/aprime - bprime*dprime/aprime/aprime/numer;
        jacobian(0,1) = bprime/aprime/numer;
        jacobian(1,0) = dprime/aprime/numer;
        jacobian(1,1) = -1/numer;
        translation[0] = xoffprime*bprime*dprime/aprime/aprime/numer - yoffprime*bprime/aprime/numer - xoffprime/aprime;
        translation[1] = yoffprime/numer - xoffprime*dprime/aprime/numer;
        affineTransform = lsst::afw::geom::AffineTransform(jacobian, translation);
        return lsst::afw::geom::AffineXYTransform(affineTransform);
    }

    lsst::afw::geom::AffineXYTransform makePixelFpTransform(lsst::afw::geom::Extent2D const pixelSizeMm, lsst::afw::geom::Extent2I const numPixels) {
        Eigen::Matrix2d jacobian;
        Eigen::Vector2d translation; 
        lsst::afw::geom::AffineTransform affineTransform;
        jacobian(0,0) = _coeffA*pixelSizeMm.getX();
        jacobian(0,1) = _coeffB*pixelSizeMm.getY();
        jacobian(1,0) = _coeffD*pixelSizeMm.getX();
        jacobian(1,1) = _coeffE*pixelSizeMm.getY();
        translation[0] = _offset.getX() - pixelSizeMm.getX()*numPixels.getX()/2.;
        translation[1] = _offset.getY() - pixelSizeMm.getY()*numPixels.getY()/2.;
        affineTransform = lsst::afw::geom::AffineTransform(jacobian, translation);
        return lsst::afw::geom::AffineXYTransform(affineTransform);
    }
private:
    lsst::afw::geom::Point2D _offset;     // offset

    lsst::afw::geom::Angle _yaw;        // yaw
    double _cosYaw;                     // cos(yaw)
    double _sinYaw;                     // sin(yaw)

    lsst::afw::geom::Angle _roll;       // roll
    double _cosRoll;                    // cos(roll)
    double _sinRoll;                    // sin(roll)

    lsst::afw::geom::Angle _pitch;      // pitch
    double _cosPitch;                   // cos(pitch)
    double _sinPitch;                   // sin(pitch)

    //Elements of the Jacobian for three space rotation projected into XY plane.
    double _coeffA;                     // 01 element
    double _coeffB;                     // 11 element
    double _coeffD;                     // 00 element
    double _coeffE;                     // 10 element
};

}}}

#endif
