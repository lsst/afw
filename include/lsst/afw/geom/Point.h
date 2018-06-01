// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2018 LSST Corporation.
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

#ifndef LSST_AFW_GEOM_POINT_H
#define LSST_AFW_GEOM_POINT_H

#include "lsst/geom/Point.h"
// preserve old lsst/afw/geom includes for backwards compatibility
#include "lsst/afw/geom/Extent.h"

namespace lsst {
namespace afw {
namespace geom {

// for backwards compatibility make lsst::geom headers and symbols available in lsst::afw::geom

using lsst::geom::Point;

using lsst::geom::PointI;
using lsst::geom::Point2I;
using lsst::geom::Point3I;

using lsst::geom::PointD;
using lsst::geom::Point2D;
using lsst::geom::Point3D;

}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_GEOM_POINT_H
