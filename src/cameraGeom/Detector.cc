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
    DetectorType type,
    std::string const &serial,
    AmplifierList const &amplifierList,
    Orientation const &orientation,
    double pixelSize,
    CameraTransformMap const &transformMap
) :
    _name(name),
    _type(type),
    _serial(serial),
    _amplifierList(amplifierList),
    _amplifierMap(),
    _orientation(orientation),
    _pixelSize(pixelSize),
    _transformRegistry(PIXELS, transformMap)
{
    _makeAmplifierMap();
}

CONST_PTR(Amplifier) Detector::operator[](std::string const &name) const {
    _AmpMap::const_iterator ampIter = _amplifierMap.find(name);
    if (ampIter == _amplifierMap.end()) {
        std::ostringstream os;
        os << "Unknown amplifier \"" << name << "\"";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    return ampIter->second;
}

 void Detector::_makeAmplifierMap() {
    for (AmplifierList::const_iterator ampIter = _amplifierList.begin();
        ampIter != _amplifierList.end(); ++ampIter) {
        _amplifierMap.insert(std::make_pair((*ampIter)->getName(), *ampIter));
    }
}

}}}
