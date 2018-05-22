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

#include "lsst/geom/SpherePoint.h"
// preserve old lsst/afw/geom includes for backwards compatibility
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"

namespace lsst {
namespace afw {
namespace geom {

using lsst::geom::SpherePoint;

using lsst::geom::averageSpherePoint;

}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif /* LSST_AFW_GEOM_SPHEREPOINT_H_ */
