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

#include <sstream>
#include <utility>
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

Detector::Detector(
    std::string const &name,
    int id,
    DetectorType type,
    std::string const &serial,
    geom::Box2I const &bbox,
    table::AmpInfoCatalog const &ampInfoCatalog,
    Orientation const &orientation,
    geom::Extent2D const & pixelSize,
    CameraTransformMap::Transforms const &transforms
) :
    _name(name),
    _id(id),
    _type(type),
    _serial(serial),
    _bbox(bbox),
    _ampInfoCatalog(ampInfoCatalog),
    _ampNameIterMap(),
    _orientation(orientation),
    _pixelSize(pixelSize),
    _transformMap(CameraSys(PIXELS.getSysName(), name), transforms)
{
    _init();
}

std::vector<geom::Point2D> Detector::getCorners(CameraSys const &cameraSys) const {
    std::vector<geom::Point2D> fromVec = geom::Box2D(_bbox).getCorners();
    return _transformMap.transform(fromVec, _transformMap.getNativeCoordSys(), cameraSys);
}

std::vector<geom::Point2D> Detector::getCorners(CameraSysPrefix const &cameraSysPrefix) const {
    return getCorners(makeCameraSys(cameraSysPrefix));
}

CameraPoint Detector::getCenter(CameraSys const &cameraSys) const {
    CameraPoint ctrPix = makeCameraPoint(geom::Box2D(_bbox).getCenter(), _transformMap.getNativeCoordSys());
    return transform(ctrPix, cameraSys);
}

CameraPoint Detector::getCenter(CameraSysPrefix const &cameraSysPrefix) const {
    return getCenter(makeCameraSys(cameraSysPrefix));
}

const table::AmpInfoRecord & Detector::operator[](std::string const &name) const {
    _AmpInfoMap::const_iterator ampIter = _ampNameIterMap.find(name);
    if (ampIter == _ampNameIterMap.end()) {
        std::ostringstream os;
        os << "Unknown amplifier \"" << name << "\"";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    return *(ampIter->second);
}

 void Detector::_init() {
    // make _ampNameIterMap
    for (table::AmpInfoCatalog::const_iterator ampIter = _ampInfoCatalog.begin();
        ampIter != _ampInfoCatalog.end(); ++ampIter) {
        _ampNameIterMap.insert(std::make_pair(ampIter->getName(), ampIter));
    }
    if (_ampNameIterMap.size() != _ampInfoCatalog.size()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
            "Invalid ampInfoCatalog: not all amplifier names are unique");
    }

    // check detector name in CoordSys in transform registry
    for (CameraTransformMap::Transforms::const_iterator trIter = _transformMap.begin();
        trIter != _transformMap.end(); ++trIter) {
            if (trIter->first.hasDetectorName() && trIter->first.getDetectorName() != _name) {
                std::ostringstream os;
                os << "Invalid transformMap: " << trIter->first << " detector name != \"" << _name << "\"";
                throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
            }
    }
}

}}}
