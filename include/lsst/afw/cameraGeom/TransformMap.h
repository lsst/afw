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
#include <unordered_map>
#include <memory>

#include "boost/iterator/transform_iterator.hpp"
#include "astshim/FrameSet.h"

#include "lsst/afw/cameraGeom/CameraSys.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/geom/Transform.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * A registry of 2-dimensional coordinate transforms for a specific camera.
 *
 * Represents the interrelationships between camera coordinate systems for a particular camera.
 * It can be seen as a mapping between a pair of CameraSys and a Transform between them
 * (though it does not conform to either the C++ map or Python dictionary APIs).
 *
 * TransformMap supports:
 * * Transforming between any two supported CameraSys using the @ref transform methods.
 * * Retrieving a transform between any two supported CameraSys using @ref getTransform.
 * * Iteration over supported CameraSys using @ref begin and @ref end in C++
 *   and standard Python iteration in Python.
 *
 * TransformMap is immutable and must always be held by shared_ptr; this is
 * enforced by making all non-deleted constructors private.  Simple
 * TransformMaps in which all transforms are relative to a single CameraSys
 * can be constructed via the `make` static member function, while more general
 * construction is provided by the Builder class.
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

    using CameraSysFrameIdMap = std::unordered_map<CameraSys, int>;

public:

    using Transforms = std::unordered_map<CameraSys, std::shared_ptr<geom::TransformPoint2ToPoint2>>;
    using CameraSysIterator = boost::transform_iterator<GetKey, CameraSysFrameIdMap::const_iterator>;

    class Builder;

    /**
     * Construct a TransformMap with all transforms relative to a single reference CameraSys.
     *
     * @param reference  Coordinate system from which each Transform in `transforms` converts.
     * @param transforms  A map whose keys are camera coordinate systems, and whose values
     *                    point to Transforms that convert from `reference` to the corresponding key.
     *                    All Transforms must be invertible.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if `transforms` contains
     *         the `reference` camera system as a key, or if any Transform is not invertible.
     */
    static std::shared_ptr<TransformMap const> make(
        CameraSys const &reference,
        Transforms const &transforms
    );

    ///@{
    /// TransformMap is immutable, so both moving and copying are prohibited.
    /// It is also always held by shared_ptr, so there is no good reason to
    /// copy it.
    TransformMap(TransformMap const &other) = delete;
    TransformMap(TransformMap &&other) = delete;
    TransformMap &operator=(TransformMap const &) = delete;
    TransformMap &operator=(TransformMap &&) = delete;
    ///@}

    ~TransformMap() noexcept;

    /**
     * Convert a point from one camera coordinate system to another.
     *
     * @param point  Point from which to transform
     * @param fromSys, toSys  Camera coordinate systems between which to transform
     * @returns the transformed value. Equivalent to
     *          `getTransform(fromSys, toSys).applyForward(point)`.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if either
     *         `fromSys` or `toSys` is not supported.
     */
    lsst::geom::Point2D transform(lsst::geom::Point2D const &point, CameraSys const &fromSys,
                                  CameraSys const &toSys) const;

    /**
     * Convert a list of points from one coordinate system to another.
     *
     * @overload
     */
    std::vector<lsst::geom::Point2D> transform(std::vector<lsst::geom::Point2D> const &pointList,
                                               CameraSys const &fromSys, CameraSys const &toSys) const;

    CameraSysIterator begin() const { return boost::make_transform_iterator(_frameIds.begin(), GetKey()); }

    CameraSysIterator end() const { return boost::make_transform_iterator(_frameIds.end(), GetKey()); }

    /**
     * Can this transform to and from the specified coordinate system?
     *
     * @param system  The coordinate system to search for
     * @returns `true` if `system` is supported, `false` otherwise
     *
     * @exceptsafe Shall not throw exceptions.
     */
    bool contains(CameraSys const &system) const noexcept;

    /**
     * Get a Transform from one camera coordinate system to another.
     *
     * @param fromSys, toSys  Camera coordinate systems between which to transform
     * @returns a Transform that converts from `fromSys` to `toSys` in the forward direction.
     *      The Transform will be invertible.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if either
     *         `fromSys` or `toSys` is not supported.
     */
    std::shared_ptr<geom::TransformPoint2ToPoint2> getTransform(CameraSys const &fromSys,
                                                                CameraSys const &toSys) const;

    /**
     * Get the number of supported coordinate systems.
     *
     * @exceptsafe Shall not throw exceptions.
     */
    size_t size() const noexcept;

private:

    // Private ctor, only called by Builder::build().
    TransformMap(std::unique_ptr<ast::FrameSet> && transforms, CameraSysFrameIdMap && frameIds);

    /**
     * The internal frame ID corresponding to a coordinate system.
     *
     * @param system  The system to convert.
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
     * @param fromSys, toSys  Coordinate systems between which to transform
     * @return an invertible Mapping that converts from `fromSys` to `toSys`
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if either
     *         `fromSys` or `toSys` is not supported.
     */
    std::shared_ptr<ast::Mapping const> _getMapping(CameraSys const &fromSys, CameraSys const &toSys) const;

    /// Allows conversions between LSST and AST data formats
    static lsst::afw::geom::Point2Endpoint _pointConverter;

    /// Stores information on all relationships between Transforms.
    std::unique_ptr<ast::FrameSet const> const _transforms;

    /**
     * Translates from LSST coordinate ID to AST frame ID.
     *
     * Must have exactly one mapping for each Frame in `transforms`.
     */
    CameraSysFrameIdMap const _frameIds;
};


/**
 * Helper class used to incrementally construct TransformMap instances.
 */
class TransformMap::Builder {
public:

    /**
     * Construct an empty builder with no transforms and only the given
     * coordinate system.
     */
    explicit Builder(CameraSys const & reference);

    ///@{
    /// Builders are copyable, movable, and assignable.
    Builder(Builder const &);
    Builder(Builder &&);  // std::vector move construct is not noexcept until C++17
    Builder & operator=(Builder const &);
    Builder & operator=(Builder &&);  // std::vector move assignment is not noexcept until C++17
    ///@}

    ~Builder() noexcept;

    /**
     * Add a new coordinate system to the builder.
     *
     * @param  fromSys   Coordinate system for the arguments to
     *                   `transform->applyForward`.
     * @param  toSys     Coordinate system for the return values of
     *                   `transform->applyForward`.
     * @param  transform Mapping from `fromSys` to `toSys`.
     *
     * @returns `*this` to enable chained calls.
     *
     * @throws pex::exceptions::InvalidParameterError  Thrown if the transform
     *     does not have forward or inverse mapping, or if `fromSys` and
     *     `toSys` are the same.
     *
     * @exceptsafe strong
     */
    Builder & connect(CameraSys const & fromSys, CameraSys const & toSys,
                      std::shared_ptr<geom::TransformPoint2ToPoint2 const> transform);

    /**
     * Add multiple connections relative to a single reference CameraSys.
     *
     * @param fromSys     Coordinate system of the arguments to
     *                    Transform::applyForward for each transform in the
     *                    given map.
     * @param transforms  A map whose keys are camera coordinate systems, and
     *                    whose values point to Transforms that convert from
     *                    `fromSys` coordinate system to the corresponding
     *                    key. All Transforms must be invertible.
     *
     * @returns `*this` to enable chained calls.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if
     *     `transforms` contains `fromSys` camera system as a key, or if
     *     any Transform is not invertible.
     *
     * @exceptsafe strong
     */
    Builder & connect(CameraSys const & fromSys, Transforms const &transforms);

    /**
     * Add multiple connections relative to a single reference CameraSys.
     *
     * @param transforms  A map whose keys are camera coordinate systems, and
     *                    whose values point to Transforms that convert from
     *                    the `reference` coordinate system (i.e. the one the
     *                    Builder was originally constructed with) to the
     *                    corresponding key. All Transforms must be
     *                    invertible.
     *
     * @returns `*this` to enable chained calls.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if
     *     `transforms` contains the `reference` camera system as a key, or if
     *     any Transform is not invertible.
     *
     * @exceptsafe strong
     */
    Builder & connect(Transforms const &transforms) {
        return connect(_reference, transforms);
    }

    /**
     * Construct a TransformMap from the connections in the builder.
     *
     * @throws pex::exceptions::InvalidParameterError  Thrown if there is no
     *     direct or indirect connection between FOCAL_PLANE and one or more
     *     coordinate systems, or there are  duplicate connections between any
     *     two systems.
     *
     * @exceptsafe strong
      */
    std::shared_ptr<TransformMap const> build() const;

private:

    struct Connection {
        mutable bool processed;
        std::shared_ptr<geom::TransformPoint2ToPoint2 const> transform;
        CameraSys fromSys;
        CameraSys toSys;
    };

    CameraSys _reference;
    std::vector<Connection> _connections;
};


}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst

#endif
