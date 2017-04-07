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

#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace geom = lsst::afw::geom;

template <typename Derived, typename T, int N>
bool geom::allclose(
    CoordinateBase<Derived,T,N> const & a,
    CoordinateBase<Derived,T,N> const & b,
    T rtol, T atol
) {
    Eigen::Array<T,N,1> diff = (a.asEigen().array() - b.asEigen().array()).abs();
    Eigen::Array<T,N,1> rhs = (0.5*(a.asEigen().array() + b.asEigen().array())).abs();
    rhs *= rtol;
    rhs += atol;
    return (diff <= rhs).all();
}

template bool geom::allclose<geom::Point2D,double,2>(
    CoordinateBase<geom::Point2D,double,2> const &,
    CoordinateBase<geom::Point2D,double,2> const &,
    double, double
);
template bool geom::allclose<geom::Point3D,double,3>(
    CoordinateBase<geom::Point3D,double,3> const &,
    CoordinateBase<geom::Point3D,double,3> const &,
    double, double
);
template bool geom::allclose<geom::Extent2D,double,2>(
    CoordinateBase<geom::Extent2D,double,2> const &,
    CoordinateBase<geom::Extent2D,double,2> const &,
    double, double
);
template bool geom::allclose<geom::Extent3D,double,3>(
    CoordinateBase<geom::Extent3D,double,3> const &,
    CoordinateBase<geom::Extent3D,double,3> const &,
    double, double
);
