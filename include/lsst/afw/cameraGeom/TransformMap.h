/*
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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

#if !defined(LSST_AFW_CAMERAGEOM_TRANSFORMMAP_H)
#define LSST_AFW_CAMERAGEOM_TRANSFORMMAP_H

#include <vector>
#include <map>
#include <unordered_map>

#include "boost/iterator/transform_iterator.hpp"
#include "astshim/FrameSet.h"

#include "lsst/afw/cameraGeom/CameraSys.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Transform.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * A registry of 2-dimensional coordinate transforms for a specific camera.
 *
 * Represents the interrelationships between camera coordinate systems for a
 * particular camera. It can be seen as a mapping between a pair of CameraSys
 * and a Transform between them, but does not conform to either the C++ map or
 * Python dictionary APIs.
 *
 * TransformMap supports transforming between any two supported CameraSys using
 * the @ref transform methods, even if no explicit transform was provided.
 *
 * @exceptsafe Unless otherwise specified, all methods guarantee only basic
 *             exception safety.
 */
class TransformMap final {
private:
     // Functor for boost::transform_iterator: given an entry in a std::map or unordered_map, return the key
    struct GetKey {
        CameraSys const &operator()(std::pair<const CameraSys, int> const &p) const { return p.first; };
    };

public:
    using Transform = afw::geom::Transform<geom::Point2Endpoint, geom::Point2Endpoint>;
    using Transforms = std::unordered_map<CameraSys, std::shared_ptr<Transform>>;
    using CameraSysFrameIdMap = std::unordered_map<CameraSys, int>;
    using CameraSysIterator = boost::transform_iterator<GetKey, CameraSysFrameIdMap::const_iterator>;

    /**
     * Define a set of camera transforms.
     *
     * @param reference Coordinate system to which `transforms` converts.
     * @param transforms A map whose keys are coordinate systems, and whose
     *                   values point to Transforms that convert from
     *                   `reference` to the corresponding key. All Transforms
     *                   must be invertible.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if
     *         `transforms` contains `reference`, or if any Transform is not
     *         invertible.
     */
    TransformMap(CameraSys const &reference,
                 std::map<CameraSys, std::shared_ptr<Transform>> const &transforms);

    /**
     * Define a set of camera transforms.
     *
     * @overload
     */
    TransformMap(CameraSys const &reference,
                 std::unordered_map<CameraSys, std::shared_ptr<Transform>> const &transforms);

    /**
     * Create a TransformMap supporting the same Transforms.
     *
     * @param other the map to copy.
     */
    TransformMap(TransformMap const &other);

    /**
     * Create a TransformMap supporting the same Transforms using move semantics.
     *
     * @param other the map to move from. It shall be left unchanged, as
     *              required by immutability.
     */
    TransformMap(TransformMap const &&other);

    ///@{
    /// TransformMap is immutable.
    TransformMap &operator=(TransformMap const &) = delete;
    TransformMap &operator=(TransformMap &&) = delete;
    ///@}

    /**
     * Convert a point from one coordinate system to another.
     *
     * @param point point from which to transform
     * @param fromSys, toCoordSys coordinate systems between which to transform
     * @returns the transformed value. Equivalent to
     *          `getTransform(fromSys, toCoordSys).applyForward(point)`.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if either
     *         `fromSys` or `toCoordSys` is not supported.
     */
    geom::Point2D transform(geom::Point2D const &point, CameraSys const &fromSys,
                            CameraSys const &toCoordSys) const;

    /**
     * Convert a list of points from one coordinate system to another.
     *
     * @overload
     */
    std::vector<geom::Point2D> transform(std::vector<geom::Point2D> const &pointList,
                                         CameraSys const &fromSys, CameraSys const &toCoordSys) const;

    CameraSysIterator begin() const { return boost::make_transform_iterator(_frameIds.begin(), GetKey()); }

    CameraSysIterator end() const { return boost::make_transform_iterator(_frameIds.end(), GetKey()); }

    /**
     * Can this transform to and from the specified coordinate system?
     *
     * @param system the coordinate system to search for
     * @returns `true` if `system` is supported, `false` otherwise
     *
     * @exceptsafe Shall not throw exceptions.
     */
    bool contains(CameraSys const &system) const noexcept;

    /**
     * Get a Transform from one camera coordinate system to another.
     *
     * @param fromSys, toSys camera coordinate systems between which to transform
     * @returns a Transform that converts from `fromSys` to `toSys` in the forward direction.
     *      The Transform will be invertible.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if either
     *         `fromSys` or `toSys` is not supported.
     */
    std::shared_ptr<Transform> getTransform(CameraSys const &fromSys, CameraSys const &toCoordSys) const;

    /**
     * Get the number of supported coordinate systems.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    size_t size() const noexcept;

private:
    /**
     * The internal frame ID corresponding to a coordinate system.
     *
     * @param system The system to convert.
     * @return the ID by which the coordinate system can be found in transforms.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if `system`
     *         is not in frameIds.
     *
     * @exceptsafe Provides strong exception guarantee.
     */
    int _getFrame(CameraSys const &system) const;

    /**
     * An ast::Mapping that transforms between two coordinate systems.
     *
     * @param fromSys, toCoordSys coordinate systems between which to transform
     * @return an invertible Mapping that converts from `fromSys` to `toCoordSys`
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if either
     *         `fromSys` or `toCoordSys` is not supported.
     */
    std::shared_ptr<ast::Mapping const> _getMapping(CameraSys const &fromSys,
                                                    CameraSys const &toCoordSys) const;

    /// Allows conversions between LSST and AST data formats
    static geom::Point2Endpoint _pointConverter;

    /// Stores information on all relationships between Transforms.
    // May be shared between multiple copies of TransformMap, since TransformMap is immutable
    std::shared_ptr<ast::FrameSet const> const _transforms;

    /**
     * Translates from LSST coordinate ID to AST frame ID.
     *
     * Must have exactly one mapping for each Frame in `transforms`.
     */
    CameraSysFrameIdMap const _frameIds;
};

}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst

#endif
