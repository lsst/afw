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

#include "lsst/afw/cameraGeom/RawAmplifier.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

    RawAmplifier::RawAmplifier(
        geom::Box2I const &bbox,
        geom::Box2I const &dataBBox,
        geom::Box2I const &horizontalOverscanBBox,
        geom::Box2I const &verticalOverscanBBox,
        geom::Box2I const &prescanBBox,
        bool flipX,
        bool flipY,
        geom::Extent2I const &xyOffset
    ) :
        Citizen(typeid(this)),
        _bbox(bbox),
        _dataBBox(dataBBox),
        _horizontalOverscanBBox(horizontalOverscanBBox),
        _verticalOverscanBBox(verticalOverscanBBox),
        _prescanBBox(prescanBBox),
        _flipX(flipX),
        _flipY(flipY),
        _xyOffset(xyOffset)
    {}

}}}
