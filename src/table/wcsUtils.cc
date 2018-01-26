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

/*
 * Get the schema from a collection of records
 *
 * This function and getValues and setValues below appear in two forms:
 * a catalog of records and a vector of shared pointers to records; the crucial difference
 * is whether elements are records or shared pointers to records
 */
template <typename Record>
Schema getSchema(CatalogT<Record> const &catalog) {
    return catalog[0].getSchema();
}

template <typename Record>
Schema getSchema(std::vector<std::shared_ptr<Record>> const &recordList) {
    return recordList[0]->getSchema();
}

// Get values for a field in a collection of catalog records
template <typename Record, typename Key>
std::vector<typename Key::Value> getValues(std::vector<std::shared_ptr<Record>> const &recordList,
                                           Key const &key) {
    std::vector<typename Key::Value> valueList;
    valueList.reserve(recordList.size());
    for (auto const &recordPtr : recordList) {
        valueList.emplace_back(recordPtr->get(key));
    }
    return valueList;
}

template <typename Record, typename Key>
std::vector<typename Key::Value> getValues(CatalogT<Record> const &catalog, Key const &key) {
    std::vector<typename Key::Value> valueList;
    valueList.reserve(catalog.size());
    for (auto const &record : catalog) {
        valueList.emplace_back(record.get(key));
    }
    return valueList;
}

/// Set values for a field in a collection of catalog records
template <typename Record, typename Key>
void setValues(std::vector<std::shared_ptr<Record>> &recordList, Key const &key,
               std::vector<typename Key::Value> const &valueList) {
    auto valuePtr = valueList.cbegin();
    for (auto &recordPtr : recordList) {
        recordPtr->set(key, *valuePtr);
        ++valuePtr;
    }
}

template <typename Record, typename Key>
void setValues(CatalogT<Record> &catalog, Key const &key, std::vector<typename Key::Value> const &valueList) {
    auto valuePtr = valueList.cbegin();
    for (auto &record : catalog) {
        record.set(key, *valuePtr);
        ++valuePtr;
    }
}

}  // namespace

template <typename ReferenceCollection>
void updateRefCentroids(geom::SkyWcs const &wcs, ReferenceCollection &refList) {
    if (refList.empty()) {
        return;
    }
    auto const schema = getSchema(refList);
    CoordKey const coordKey(schema["coord"]);
    Point2DKey const centroidKey(schema["centroid"]);
    std::vector<coord::IcrsCoord> const skyList = getValues(refList, coordKey);
    std::vector<geom::Point2D> const pixelList = wcs.skyToPixel(skyList);
    setValues(refList, centroidKey, pixelList);
}

template <typename SourceCollection>
void updateSourceCoords(geom::SkyWcs const &wcs, SourceCollection &sourceList) {
    if (sourceList.empty()) {
        return;
    }
    auto const schema = getSchema(sourceList);
    Point2DKey const centroidKey(schema["slot_Centroid"]);
    CoordKey const coordKey(schema["coord"]);
    std::vector<geom::Point2D> pixelList = getValues(sourceList, centroidKey);
    std::vector<coord::IcrsCoord> const skyList = wcs.pixelToSky(pixelList);
    setValues(sourceList, coordKey, skyList);
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
