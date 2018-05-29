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

#include <memory>
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/wcsUtils.h"

namespace lsst {
namespace afw {
namespace table {
namespace {

// Get the schema from a record
Schema getSchema(BaseRecord const &record) { return record.getSchema(); }

// Get the schema from a shared pointer to a record
Schema getSchema(std::shared_ptr<const BaseRecord> const record) { return record->getSchema(); }

// Get a value from a record
template <typename Key>
inline typename Key::Value getValue(BaseRecord const &record, Key const &key) {
    return record.get(key);
}

// Get a value from a shared pointer to a record
template <typename Key>
inline typename Key::Value getValue(std::shared_ptr<const BaseRecord> const record, Key const &key) {
    return record->get(key);
}

// Set a value in a record
template <typename Key>
inline void setValue(SimpleRecord &record, Key const &key, typename Key::Value const &value) {
    record.set(key, value);
}

// Set a value in a shared pointer to a record
template <typename Key>
inline void setValue(std::shared_ptr<SimpleRecord> record, Key const &key, typename Key::Value const &value) {
    record->set(key, value);
}

}  // namespace

template <typename ReferenceCollection>
void updateRefCentroids(geom::SkyWcs const &wcs, ReferenceCollection &refList) {
    if (refList.empty()) {
        return;
    }
    auto const schema = getSchema(refList[0]);
    CoordKey const coordKey(schema["coord"]);
    Point2DKey const centroidKey(schema["centroid"]);
    Key<Flag> const hasCentroidKey(schema["hasCentroid"]);
    std::vector<SpherePoint> skyList;
    skyList.reserve(refList.size());
    for (auto const &record : refList) {
        skyList.emplace_back(getValue(record, coordKey));
    }
    std::vector<geom::Point2D> const pixelList = wcs.skyToPixel(skyList);
    auto pixelPos = pixelList.cbegin();
    for (auto &refObj : refList) {
        setValue(refObj, centroidKey, *pixelPos);
        setValue(refObj, hasCentroidKey, true);
        ++pixelPos;
    }
}

template <typename SourceCollection>
void updateSourceCoords(geom::SkyWcs const &wcs, SourceCollection &sourceList) {
    if (sourceList.empty()) {
        return;
    }
    auto const schema = getSchema(sourceList[0]);
    Point2DKey const centroidKey(schema["slot_Centroid"]);
    CoordKey const coordKey(schema["coord"]);
    std::vector<geom::Point2D> pixelList;
    pixelList.reserve(sourceList.size());
    for (auto const &source : sourceList) {
        pixelList.emplace_back(getValue(source, centroidKey));
    }
    std::vector<SpherePoint> const skyList = wcs.pixelToSky(pixelList);
    auto skyCoord = skyList.cbegin();
    for (auto &source : sourceList) {
        setValue(source, coordKey, *skyCoord);
        ++skyCoord;
    }
}

/// @cond
template void updateRefCentroids(geom::SkyWcs const &, std::vector<std::shared_ptr<SimpleRecord>> &);
template void updateRefCentroids(geom::SkyWcs const &, SimpleCatalog &);

template void updateSourceCoords(geom::SkyWcs const &, std::vector<std::shared_ptr<SourceRecord>> &);
template void updateSourceCoords(geom::SkyWcs const &, SourceCatalog &);
/// @endcond

}  // namespace table
}  // namespace afw
}  // namespace lsst
