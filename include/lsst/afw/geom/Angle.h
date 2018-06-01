// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2018 LSST Corporation.
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
#ifndef LSST_AFW_GEOM_ANGLE_H
#define LSST_AFW_GEOM_ANGLE_H

#include "lsst/geom/Angle.h"

namespace lsst {
namespace afw {
namespace geom {

// for backwards compatibility make lsst::geom headers and symbols available in lsst::afw::geom

using lsst::geom::Angle;
using lsst::geom::AngleUnit;

using lsst::geom::arcminutes;
using lsst::geom::arcseconds;
using lsst::geom::degrees;
using lsst::geom::hours;
using lsst::geom::radians;

using lsst::geom::HALFPI;
using lsst::geom::INVSQRTPI;
using lsst::geom::ONE_OVER_PI;
using lsst::geom::PI;
using lsst::geom::ROOT2;
using lsst::geom::SQRTPI;
using lsst::geom::TWOPI;

using lsst::geom::arcsecToRad;
using lsst::geom::degToRad;
using lsst::geom::masToRad;
using lsst::geom::radToArcsec;
using lsst::geom::radToDeg;
using lsst::geom::radToMas;

}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_GEOM_ANGLE_H
