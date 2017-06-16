// -*- lsst-c++ -*-

#ifndef LSST_AFW_GEOM_TRANSFORMFACTORY_H
#define LSST_AFW_GEOM_TRANSFORMFACTORY_H

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

/*
 * Functions for producing Transforms with commonly desired properties.
 */

#include "lsst/afw/geom/Transform.h"

namespace lsst {
namespace afw {
namespace geom {

/**
 * Approximate a Transform by its local linearization.
 *
 * @tparam FromEndpoint, ToEndpoint The endpoints of the transform.
 *
 * @param original the Transform to linearize
 * @param point the point at which a linear approximation is desired
 * @returns a linear Transform whose value and Jacobian at `point` match those
 *         of `original`.
 *
 * @throws pex::exceptions::InvalidParameterError Thrown if `original` does not
 *             have a well-defined value and Jacobian at `point`
 * @exceptsafe Provides basic exception safety.
 */
template <class FromEndpoint, class ToEndpoint>
Transform<FromEndpoint, ToEndpoint> linearizeTransform(
        Transform<FromEndpoint, ToEndpoint> const &original,
        typename Transform<FromEndpoint, ToEndpoint>::FromPoint const &point);

/*
 * The correct behavior for linearization is unclear where SpherePoints are involved (see discussion on
 * DM-10542). Forbid usage until somebody needs it. Note to maintainers: the template specializations MUST
 * be deleted in the header for compilers to complain correctly.
 */
#define DISABLE(From, To)                                                         \
    template <>                                                                   \
    Transform<From, To> linearizeTransform<From, To>(Transform<From, To> const &, \
                                                     Transform<From, To>::FromPoint const &) = delete;
DISABLE(GenericEndpoint, SpherePointEndpoint);
DISABLE(Point2Endpoint, SpherePointEndpoint);
DISABLE(SpherePointEndpoint, GenericEndpoint);
DISABLE(SpherePointEndpoint, Point2Endpoint);
DISABLE(SpherePointEndpoint, SpherePointEndpoint);
#undef DISABLE

}  // geom
}  // afw
}  // lsst

#endif  // LSST_AFW_GEOM_TRANSFORMFACTORY_H
