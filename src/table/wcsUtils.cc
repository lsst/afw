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

#include <cmath>
#include <memory>
#include <vector>

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
    std::vector<lsst::geom::SpherePoint> skyList;
    skyList.reserve(refList.size());
    for (auto const &record : refList) {
        skyList.emplace_back(getValue(record, coordKey));
    }
    std::vector<lsst::geom::Point2D> const pixelList = wcs.skyToPixel(skyList);
    auto pixelPos = pixelList.cbegin();
    for (auto &refObj : refList) {
        setValue(refObj, centroidKey, *pixelPos);
        setValue(refObj, hasCentroidKey, true);
        ++pixelPos;
    }
}

Eigen::Matrix2f calculateCoordCovariance(geom::SkyWcs const &wcs, lsst::geom::Point2D center,
                                         Eigen::Matrix2f err) {
    if (!std::isfinite(center.getX()) || !std::isfinite(center.getY())) {
        return Eigen::Matrix2f::Constant(NAN);
    }
    // Get the derivative of the pixel-to-sky transformation, then use it to
    // propagate the centroid uncertainty to coordinate uncertainty. Note that
    // the calculation is done in arcseconds, then converted to radians in
    // order to achieve higher precision.
    const static double scale = 1.0 / 3600.0;
    const static Eigen::Matrix2d cdMatrix{{scale, 0}, {0, scale}};

    lsst::geom::SpherePoint skyCenter = wcs.pixelToSky(center);
    auto localGnomonicWcs = geom::makeSkyWcs(center, skyCenter, cdMatrix);
    auto measurementToLocalGnomonic = wcs.getTransform()->then(*localGnomonicWcs->getTransform()->inverted());

    Eigen::Matrix2d localMatrix = measurementToLocalGnomonic->getJacobian(center);
    Eigen::Matrix2f d = localMatrix.cast<float>() * scale * (lsst::geom::PI / 180.0);

    Eigen::Matrix2f skyCov = d * err * d.transpose();

    // Multiply by declination correction matrix in order to get sigma(RA) * cos(Dec) for the uncertainty
    // in RA, and cov(RA, Dec) * cos(Dec) for the RA/Dec covariance:
    float cosDec = std::cos(skyCenter.getDec().asRadians());
    Eigen::Matrix2f decCorr{{cosDec, 0}, {0, 1.0}};
    Eigen::Matrix2f skyCovCorr = decCorr * skyCov * decCorr;
    return skyCovCorr;
}

template <typename SourceCollection>
void updateSourceCoords(geom::SkyWcs const &wcs, SourceCollection &sourceList, bool include_covariance) {
    if (sourceList.empty()) {
        return;
    }
    auto const schema = getSchema(sourceList[0]);
    Point2DKey const centroidKey(schema["slot_Centroid"]);
    CovarianceMatrixKey<float, 2> const centroidErrKey(schema["slot_Centroid"], {"x", "y"});
    CoordKey const coordKey(schema["coord"]);
    std::vector<lsst::geom::Point2D> pixelList;
    pixelList.reserve(sourceList.size());
    std::vector<Eigen::Matrix2f> skyErrList;
    skyErrList.reserve(sourceList.size());

    for (auto const &source : sourceList) {
        lsst::geom::Point2D center = getValue(source, centroidKey);
        pixelList.emplace_back(center);
        if (include_covariance) {
            auto err = getValue(source, centroidErrKey);
            Eigen::Matrix2f skyCov = calculateCoordCovariance(wcs, center, err);
            skyErrList.emplace_back(skyCov);
        }
    }

    std::vector<lsst::geom::SpherePoint> const skyList = wcs.pixelToSky(pixelList);
    auto skyCoord = skyList.cbegin();
    auto skyErr = skyErrList.cbegin();
    for (auto &source : sourceList) {
        setValue(source, coordKey, *skyCoord);
        if (include_covariance) {
            CoordKey::ErrorKey const coordErrKey = CoordKey::getErrorKey(schema);
            setValue(source, coordErrKey, *skyErr);
        }
        ++skyCoord;
        ++skyErr;
    }
}

/// @cond
template void updateRefCentroids(geom::SkyWcs const &, std::vector<std::shared_ptr<SimpleRecord>> &);
template void updateRefCentroids(geom::SkyWcs const &, SimpleCatalog &);

template void updateSourceCoords(geom::SkyWcs const &, std::vector<std::shared_ptr<SourceRecord>> &,
                                 bool include_covariance);
template void updateSourceCoords(geom::SkyWcs const &, SourceCatalog &, bool include_covariance);
/// @endcond

}  // namespace table
}  // namespace afw
}  // namespace lsst
