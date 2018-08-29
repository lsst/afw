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

#if !defined(LSST_AFW_CAMERAGEOM_ORIENTATION_H)
#define LSST_AFW_CAMERAGEOM_ORIENTATION_H

#include <string>
#include <cmath>
#include "Eigen/Dense"
#include "lsst/geom.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/image/Utils.h"

/*
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
class Orientation final {
public:
    explicit Orientation(lsst::geom::Point2D const fpPosition = lsst::geom::Point2D(0, 0),
                         ///< Focal plane position of detector reference point (mm)
                         lsst::geom::Point2D const refPoint = lsst::geom::Point2D(-0.5, -0.5),
                         ///< Reference point on detector (pixels).
                         ///< Offset is measured to this point and all all rotations are about this point.
                         ///< The default value (-0.5, -0.5) is the lower left corner of the detector.
                         lsst::geom::Angle const yaw =
                                 lsst::geom::Angle(0),  ///< yaw: rotation about Z (X to Y), 1st rotation
                         lsst::geom::Angle const pitch = lsst::geom::Angle(
                                 0),  ///< pitch: rotation about Y' (Z'=Z to X'), 2nd rotation
                         lsst::geom::Angle const roll = lsst::geom::Angle(
                                 0)  ///< roll: rotation about X'' (Y''=Y' to Z''), 3rd rotation
    );

    ~Orientation() noexcept;
    Orientation(Orientation const &) noexcept;
    Orientation(Orientation &&) noexcept;
    Orientation &operator=(Orientation const &) noexcept;
    Orientation &operator=(Orientation &&) noexcept;

    /// Return focal plane position of detector reference point (mm)
    lsst::geom::Point2D getFpPosition() const noexcept { return _fpPosition; }

    /// Return detector reference point (pixels)
    lsst::geom::Point2D getReferencePoint() const noexcept { return _refPoint; }

    /// Return the yaw angle
    lsst::geom::Angle getYaw() const noexcept { return _yaw; }

    /// Return the pitch angle
    lsst::geom::Angle getPitch() const noexcept { return _pitch; }

    /// Return the roll angle
    lsst::geom::Angle getRoll() const noexcept { return _roll; }

    /// Return the number of quarter turns (rounded to the closest quarter)
    int getNQuarter() const noexcept;

    /**
     * Generate a Transform from pixel to focal plane coordinates
     *
     * @returns lsst::afw::geom::Transform from pixel to focal plane coordinates
     */
    std::shared_ptr<geom::TransformPoint2ToPoint2> makePixelFpTransform(
            lsst::geom::Extent2D const pixelSizeMm  ///< Size of the pixel in mm in X and Y
            ) const;

    /**
     * Generate a Transform from focal plane to pixel coordinates
     *
     * @returns lsst::afw::geom::Transform from focal plane to pixel coordinates
     */
    std::shared_ptr<geom::TransformPoint2ToPoint2> makeFpPixelTransform(
            lsst::geom::Extent2D const pixelSizeMm  ///< Size of the pixel in mm in X and Y
            ) const;

private:
    lsst::geom::Point2D _fpPosition;  ///< focal plane position of reference point on detector
    lsst::geom::Point2D _refPoint;    ///< reference point on detector

    lsst::geom::Angle _yaw;    ///< yaw
    lsst::geom::Angle _pitch;  ///< pitch
    lsst::geom::Angle _roll;   ///< roll

    // Elements of the Jacobian for three space rotation projected into XY plane.
    // Turn off alignment since this is dynamically allocated (via Detector)
    Eigen::Matrix<double, 2, 2, Eigen::DontAlign> _rotMat;
};
}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst

#endif
