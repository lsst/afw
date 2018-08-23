/*
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "lsst/afw/cameraGeom/DetectorCollection.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

DetectorCollection::DetectorCollection(List const & detectorList) {
    for (auto const & detector : detectorList) {
        _nameDict[detector->getName()] = detector;
        _idDict[detector->getId()] = detector;
        for (auto const & corner : detector->getCorners(FOCAL_PLANE)) {
            _fpBBox.include(corner);
        }
    }

    if (_idDict.size() < detectorList.size()) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Detector IDs are not unique");
    }
    if (_nameDict.size() < detectorList.size()) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Detector names are not unique");
    }
}

DetectorCollection::~DetectorCollection() noexcept = default;

std::shared_ptr<Detector> DetectorCollection::operator[](std::string const & name) const {
    auto det = get(name);
    if (det == nullptr) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "Detector name not found");
    }
    return det;
}

std::shared_ptr<Detector> DetectorCollection::operator[](int id) const {
    auto det = get(id);
    if (det == nullptr) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "Detector id not found");
    }
    return det;
}

std::shared_ptr<Detector> DetectorCollection::get(std::string const & name,
                                                  std::shared_ptr<Detector> def) const {
    auto i = _nameDict.find(name);
    if (i == _nameDict.end()) {
        return def;
    }
    return i->second;
}

std::shared_ptr<Detector> DetectorCollection::get(int id, std::shared_ptr<Detector> def) const {
    auto i = _idDict.find(id);
    if (i == _idDict.end()) {
        return def;
    }
    return i->second;
}

} // namespace cameraGeom
} // namespace afw
} // namespace lsst



