/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011, 2012, 2013, 2014 LSST Corporation.
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

#include "lsst/afw/geom/transformFactory.h"
#include "lsst/afw/cameraGeom/Orientation.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

Orientation::Orientation(lsst::geom::Point2D const fpPosition, lsst::geom::Point2D const refPoint,
                         lsst::geom::Angle const yaw, lsst::geom::Angle const pitch,
                         lsst::geom::Angle const roll)
        : _fpPosition(fpPosition), _refPoint(refPoint), _yaw(yaw), _pitch(pitch), _roll(roll), _rotMat() {
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
    _rotMat << cosYaw * cosPitch, cosYaw * sinPitch * sinRoll - cosRoll * sinYaw, cosPitch * sinYaw,
            cosYaw * cosRoll + sinYaw * sinPitch * sinRoll;
}

Orientation::~Orientation() noexcept = default;
// Can't combine noexcept and =default until Eigen::Matrix supports noexcept
Orientation::Orientation(Orientation const &other) noexcept
        : _fpPosition(other._fpPosition),
          _refPoint(other._refPoint),
          _yaw(other._yaw),
          _pitch(other._pitch),
          _roll(other._roll),
          _rotMat(other._rotMat) {}
Orientation::Orientation(Orientation &&other) noexcept
        : _fpPosition(std::move(other._fpPosition)),
          _refPoint(std::move(other._refPoint)),
          _yaw(std::move(other._yaw)),
          _pitch(std::move(other._pitch)),
          _roll(std::move(other._roll)),
          _rotMat(std::move(other._rotMat)) {}
Orientation &Orientation::operator=(Orientation const &other) noexcept {
    _fpPosition = other._fpPosition;
    _refPoint = other._refPoint;
    _yaw = other._yaw;
    _pitch = other._pitch;
    _roll = other._roll;
    _rotMat = other._rotMat;
    return *this;
}
Orientation &Orientation::operator=(Orientation &&other) noexcept {
    _fpPosition = std::move(other._fpPosition);
    _refPoint = std::move(other._refPoint);
    _yaw = std::move(other._yaw);
    _pitch = std::move(other._pitch);
    _roll = std::move(other._roll);
    _rotMat = std::move(other._rotMat);
    return *this;
}

int Orientation::getNQuarter() const noexcept {
    float yawDeg = _yaw.asDegrees();
    while (yawDeg < 0.) {
        yawDeg += 360.;
    }
    return std::floor((yawDeg + 45.) / 90.);
}

std::shared_ptr<afw::geom::TransformPoint2ToPoint2> Orientation::makePixelFpTransform(
        lsst::geom::Extent2D const pixelSizeMm) const {
    // jacobian = coeffA*pixelSizeMmX, coeffB*pixelSizeMmY,
    //            coeffD*pixelSizeMmX, coeffE*pixelSizeMmY
    Eigen::Matrix2d jacobian =
            _rotMat.array() * (Eigen::Vector2d::Ones() * pixelSizeMm.asEigen().transpose()).array();

    Eigen::Vector2d refMm = pixelSizeMm.asEigen().array() * _refPoint.asEigen().array();
    Eigen::Vector2d translation = _fpPosition.asEigen() - (_rotMat * refMm);

    lsst::geom::AffineTransform affineTransform = lsst::geom::AffineTransform(jacobian, translation);
    return afw::geom::makeTransform(affineTransform);
}

std::shared_ptr<afw::geom::TransformPoint2ToPoint2> Orientation::makeFpPixelTransform(
        lsst::geom::Extent2D const pixelSizeMm) const {
    return makePixelFpTransform(pixelSizeMm)->inverted();
}

}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
