// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2017 LSST Corporation.
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

#ifndef LSST_AFW_GEOM_TRANSFORM_H
#define LSST_AFW_GEOM_TRANSFORM_H

#include <memory>
#include <vector>

#include "astshim.h"
#include "ndarray.h"

#include "lsst/afw/geom/Endpoint.h"

namespace lsst {
namespace afw {
namespace geom {

/**
Transform LSST spatial data, such as Point2D and SpherePoint, using an AST transform.

This class contains two Endpoints, to specify the "from" and "to" LSST data type,
and an astshim::FrameSet or astshim::Mapping to specify the transformation.
In the case of a FrameSet the transformation is from the `BASE` frame to the `CURRENT` frame.
The endpoints convert the data between the LSST Form (e.g. Point2D) and the form used by astshim.

@note You gain some safety by constructing a Transform from an astshim::FrameSet,
since the base and current frames in the FrameSet can be checked against by the appropriate endpoint.

@note "In place" versions of `tranForward` and `tranInverse` are not available
because data must be copied when converting from LSST data types to the type used by astshim,
so it didn't seem worth the bother.
*/
template <typename FromEndpoint, typename ToEndpoint>
class Transform {
public:
    using FromArray = typename FromEndpoint::Array;
    using FromPoint = typename FromEndpoint::Point;
    using ToArray = typename ToEndpoint::Array;
    using ToPoint = typename ToEndpoint::Point;

    Transform(Transform const &) = delete;
    Transform(Transform &&) = default;
    Transform & operator=(Transform const &) = delete;
    Transform & operator=(Transform &&) = default;

    /**
    Construct a Transform from a deep copy of an ast::Mapping

    The internal FrameSet consists of a frame constructed by each endpoint
    connected by the mapping.

    @param[in] mapping  ast::Mapping describing the desired transformation
    @param[in] simplify  Simplify the mapping? This combines component mappings
        and removes redundant components where possible.
    */
    explicit Transform(ast::Mapping const &mapping, bool simplify=true);

    /**
    Constructor a Transform from a deep copy of a FrameSet.

    The result transforms from the "base" frame to the "current" frame.
    The "from" endpoint is used to normalize the "base" frame
    and the "to" endpoint is used to normalize the "current" frame.

    This is pickier than the constructor that takes an ast::Mapping in that:
    - SphereEndpoint must be associated with an ast::SkyFrame and the SkyFrame axes
      are swapped if necessary to the standard order: longitude, latitude.
    - Point2Endpoint must be associated with an ast::Frame (not a subclass),
      because Frame is the only kind of Frame that is sure to be Cartesian.

    @param[in] frameSet  ast::FrameSet describing the desired transformation in the usual way:
                         from "base" frame to "current" frame
    @param[in] simplify  Simplify the frame set? This simplifies each mapping
                         in the frame set by combining component mappings and removing
                         redundant components where possible. However it
                         does not remove any frames.
    */
    explicit Transform(ast::FrameSet const & frameSet, bool simplify=true);

    ~Transform(){};

    /**
    Get the "from" endpoint
    */
    FromEndpoint getFromEndpoint() const { return _fromEndpoint; }

    /**
    Get the contained frameset
    */
    std::shared_ptr<const ast::FrameSet> getFrameSet() const { return _frameSet; }

    /**
    Get the "to" endpoint
    */
    ToEndpoint getToEndpoint() const { return _toEndpoint; }

    /**
    Transform one point in the forward direction ("from" to "to")
    */
    ToPoint tranForward(FromPoint const & point) const;

    /**
    Transform an array of points in the forward direction ("from" to "to")
    */
    ToArray tranForward(FromArray const &array) const;

    /**
    Transform one point in the inverse direction ("to" to "from")
    */
    FromPoint tranInverse(ToPoint const & point) const;

    /**
    Transform an array of points in the inverse direction ("to" to "from")
    */
    FromArray tranInverse(ToArray const & array) const;

private:
    FromEndpoint const _fromEndpoint;
    std::shared_ptr<const ast::FrameSet> _frameSet;
    ToEndpoint const _toEndpoint;
};

/**
Print a Transform to an ostream

The format is "Transform<_fromEndpoint_, _toEndpoint_>"
where _fromEndpoint_ and _toEndpoint_ are the appropriate endpoint printed to the ostream;
for example "Transform<GenericEndpoint(4), Point3Endpoint()>"
*/
template <typename FromEndpoint, typename ToEndpoint>
std::ostream & operator<<(std::ostream & os, Transform<FromEndpoint, ToEndpoint> const & transform);

}  // geom
}  // afw
}  // lsst

#endif
