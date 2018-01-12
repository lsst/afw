// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_GEOM_SPHEREPOINT_H_
#define LSST_AFW_GEOM_SPHEREPOINT_H_

#include <ostream>
#include <utility>

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"

namespace lsst {
namespace afw {
namespace geom {

/**
 * Point in an unspecified spherical coordinate system.
 *
 * This class represents a point on a sphere in the mathematical rather
 * than the astronomical sense. It does not refer to any astronomical
 * reference frame, nor does it have any concept of (radial) distance.
 *
 * Points can be represented either as spherical coordinates or as a unit
 * vector. The adopted convention for converting between these two systems
 * is that (0, 0) corresponds to <1, 0, 0>, that (&pi;/2, 0) corresponds
 * to <0, 1, 0>, and that (0, &pi;/2) corresponds to <0, 0, 1>.
 *
 * To keep usage simple, SpherePoint does not support modification of existing
 * values; transformations of SpherePoints should be expressed as a new object.
 * To support cases where a SpherePoint *must* be updated in-place, SpherePoint
 * supports assignment to an existing object. One example is in containers
 * of SpherePoints; no STL container has an atomic element-replacement method,
 * so complicated constructions would need to be used if you couldn't
 * overwrite an existing element.
 *
 * @see @ref coord::Coord
 */
class SpherePoint final {
public:
    /**
     * Construct a SpherePoint from a longitude and latitude.
     *
     * @param longitude The longitude of the point. `+/-inf` is treated as `nan`
     *                  (because longitude is wrapped to a standard range and infinity cannot be wrapped)
     * @param latitude The latitude of the point. Must be in the
     *                 interval [-&pi;/2, &pi;/2] radians.
     *
     * @throws pex::exceptions::InvalidParameterError
     *         Thrown if `latitude` isout of range.
     *
     * @exceptsafe Provides strong exception guarantee.
     */
    SpherePoint(Angle const& longitude, Angle const& latitude);

    /**
     * Construct a SpherePoint from double longitude and latitude.
     *
     * @param longitude The longitude of the point. `+/-inf` is treated as `nan`
     *                  (because longitude is wrapped to a standard range and infinity cannot be wrapped)
     * @param latitude The latitude of the point. Must be in the
     *                 interval [-&pi;/2, &pi;/2] radians.
     * @param units The units of longitude and latitude
     *
     * @throws pex::exceptions::InvalidParameterError
     *         Thrown if `latitude` isout of range.
     *
     * @exceptsafe Provides strong exception guarantee.
     */
    SpherePoint(double longitude, double latitude, AngleUnit units);

    /**
     * Construct a SpherePoint from a vector representing a direction.
     *
     * @param vector A position whose direction will be stored as a SpherePoint.
     *               Must not be the zero vector. Need not be normalized,
     *               and the norm will not affect the value of the point.
     *
     * @throws pex::exceptions::InvalidParameterError
     *         Thrown if `vector` is the zero vector.
     *
     * @exceptsafe Provides strong exception guarantee.
     *
     * @note If the SpherePoint is at a pole then longitude is set to 0.
     * That provides predictable behavior for @ref bearingTo and @ref offset.
     */
    explicit SpherePoint(Point3D const& vector);

    /**
     * Create a copy of a SpherePoint.
     *
     * @param other The point to copy.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    SpherePoint(SpherePoint const& other) noexcept;

    /**
     * @copybrief SpherePoint(SpherePoint const&)
     *
     * As SpherePoint(SpherePoint const&), except that `other` may be left
     * in an unspecified but valid state.
     */
    SpherePoint(SpherePoint&& other) noexcept;

    ~SpherePoint();

    /**
     * Overwrite this object with the value of another SpherePoint.
     *
     * This is the only method that can alter the state of a SpherePoint after
     * its construction, and is provided to allow in-place replacement of
     * SpherePoints where swapping is not possible.
     *
     * @param other The object with which to overwrite this one.
     * @returns a reference to this object.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    SpherePoint& operator=(SpherePoint const& other) noexcept;

    /**
     * @copybrief operator=(SpherePoint const&)
     *
     * As operator=(SpherePoint const&), except that `other` may be left
     * in an unspecified but valid state.
     */
    SpherePoint& operator=(SpherePoint&& other) noexcept;

    /*
     * Accessors
     */

    /**
     * The longitude of this point.
     *
     * @returns the longitude, in the interval [0, 2&pi;) radians.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    Angle getLongitude() const noexcept { return _longitude * radians; };

    /**
     * The latitude of this point.
     *
     * @returns the latitude, in the interval [-&pi;/2, &pi;/2] radians.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    Angle getLatitude() const noexcept { return _latitude * radians; };

    /**
     * A unit vector representation of this point.
     *
     * @returns a unit vector whose direction corresponds to this point
     *
     * @exceptsafe Shall not throw exceptions.
     */
    Point3D getVector() const noexcept;

    /**
     * Longitude and latitude by index.
     *
     * @param index the index of the spherical coordinate to return. Must
     *              be either 0 or 1.
     *
     * @returns @ref getLongitude() if `index` = 0, or @ref getLatitude()
     *         if `index` = 1
     *
     * @throws pex::exceptions::OutOfRangeError
     *         Thrown if `index` is neither 0 nor 1.
     *
     * @exceptsafe Provides strong exception guarantee.
     */
    Angle operator[](size_t index) const;

    /**
     * `true` if this point is either coordinate pole.
     *
     * @returns `true` if this point is at the north or south pole,
     *         `false` otherwise
     *
     * @exceptsafe Shall not throw exceptions.
     */
    bool atPole() const noexcept {
        // Unit tests indicate we don't need to worry about epsilon-errors, in that
        // Objects constructed from lat=90*degrees, <0,0,1>, etc. all have atPole() = true.
        return fabs(_latitude) >= HALFPI;
    }

    /**
     * `true` if this point is a well-defined position.
     *
     * @returns `true` if @ref getLongitude(), @ref getLatitude(), and
     *         @ref getVector() return finite floating-point values;
     *         `false` if any return NaN or infinity.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    bool isFinite() const noexcept;

    /*
     * Comparisons between points
     */

    /**
     * `true` if two points have the same longitude and latitude.
     *
     * @param other the point to test for equality
     * @returns true if this point has exactly the same values of getLongitude() and getLatitude()
     * as `other`, false otherwise
     *
     * @exceptsafe Shall not throw exceptions.
     * 
     * @note Two points at the same pole will compare unequal if they have different longitudes,
     *      despite representing the same point on the unit sphere.
     *      This is important because the behavior of @ref offset and @ref bearingTo
     *      depend on longitude, even at the pole.
     *
     * @warning Points whose @ref getLongitude(), @ref getLatitude(), or
     *          @ref getVector() methods return `NaN` shall not compare
     *          equal to any point, including themselves. This may break
     *          algorithms that assume object equality is reflexive; use
     *          @ref isFinite() to filter objects if necessary.
     */
    bool operator==(SpherePoint const& other) const noexcept;

    /**
     * `false` if two points represent the same position.
     *
     * This operator shall always return the logical negation of operator==();
     * see its documentation for a detailed specification.
     */
    bool operator!=(SpherePoint const& other) const noexcept;

    /**
     * Orientation at this point of the great circle arc to another point.
     *
     * The orientation is measured in a plane tangent to the sphere at this point, with 0 degrees along
     * the direction of increasing longitude and 90 degrees along the direction of increasing latitude.
     * Thus for any point except the pole, the arc is due east at 0 degrees and due north at 90 degrees.
     * To understand the behavior at the poles it may help to imagine the point has been shifted slightly
     * by decreasing the absolute value of its latitude.
     *
     * If the points are on opposite sides of the sphere then the returned bearing may be any value.
     *
     * @param other the point to which to measure the bearing
     * @returns the direction, as defined above, in the interval [0, 2&pi;).
     *
     * @exceptsafe Provides strong exception guarantee.
     *
     * @note For two points `A` and `B`, `A.bearingTo(B)` will in
     *       general not be 180 degrees away from `B.bearingTo(A)`.
     */
    Angle bearingTo(SpherePoint const& other) const;

    /**
     * Angular distance between two points.
     *
     * @param other the point to which to measure the separation
     * @returns the length of the shortest (great circle) arc between the
     *         two points. Shall not be negative.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    Angle separation(SpherePoint const& other) const noexcept;

    /*
     * Transformations of points
     */

    /**
     * Return a point rotated from this one around an axis.
     *
     * @param axis a point defining the north pole of the rotation axis.
     * @param amount the amount of rotation, where positive values
     *               represent right-handed rotations around `axis`.
     * @returns a new point created by rotating this point. If the new point is at the pole
     * then its longitude will be 0.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    SpherePoint rotated(SpherePoint const& axis, Angle const& amount) const noexcept;

    /**
     * Return a point offset from this one along a great circle.
     *
     * For any point `A`, any angle `bearing` and any non-negative angle `amount`,
     * if `B` = A.offset(bearing, amount)` then `A.bearingTo(B) = amount` and `A.separationTo(B) = amount`.
     * Negative values of `amount` are supported in the obvious manner:
     * `A.offset(bearing, delta) = A.offset(bearing + 180*degrees, -delta)`
     *
     * @param bearing the direction in which to move this point, following
     *                the conventions described in @ref bearingTo.
     * @param amount the distance by which to move along the great circle defined by `bearing`
     * @returns a new point created by rotating this point. If the new point is at the pole
     * then its longitude will be 0.
     *
     * @exceptsafe Provides strong exception guarantee.
     */
    SpherePoint offset(Angle const& bearing, Angle const& amount) const;

private:
    double _longitude;  // radians
    double _latitude;   // radians
};

/*
 * Object-level display
 */

/**
 * Print the value of a point to a stream.
 *
 * The exact details of the string representation are unspecified and
 * subject to change, but the following may be regarded as typical:
 * `"(10.543250, +32.830583)"`.
 *
 * @param os the stream to which to print `point`
 * @param point the point to print to the stream
 * @returns a reference to `os`
 *
 * @throws pex::exceptions::std::ostream::failure
 *      Thrown if an I/O state flag was set that was registered with
 *      `os.exceptions()`. See the documentation of std::ostream for more
 *      details.
 *
 * @exceptsafe Provides basic exception guarantee.
 *
 * @relatesalso SpherePoint
 */
std::ostream& operator<<(std::ostream& os, SpherePoint const& point);
}
}
} /* namespace lsst::afw::geom */

#endif /* LSST_AFW_GEOM_SPHEREPOINT_H_ */
