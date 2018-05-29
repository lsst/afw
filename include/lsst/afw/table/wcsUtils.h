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

#ifndef LSST_AFW_TABLE_WCSUTILS_H
#define LSST_AFW_TABLE_WCSUTILS_H

#include "lsst/afw/geom/SkyWcs.h"

namespace lsst {
namespace afw {
namespace table {

/**
 * Update centroids in a collection of reference objects
 *
 * @tparam RefCollection  Type of sequence of reference objects, e.g. lsst::afw::table::SimpleCatalog or
 *          std::vector<std::shared_ptr<lsst::afw::table::SimpleRecord>>
 * @param[in] wcs  WCS to map from sky to pixels
 * @param[in,out] refList  Collection of reference objects. The schema must have three fields:
 *                  - "coord": a field containing an ICRS lsst::afw::SpherePoint; this field is read
 *                  - "centroid": a field containing lsst::afw::geom::Point2D; this field is written
 *                  - "hasCentroid": a flag; this field is written
 *
 * @throws lsst::pex::exceptions::NotFoundError if refList's schema does not have the required fields.
 */
template <typename ReferenceCollection>
void updateRefCentroids(geom::SkyWcs const& wcs, ReferenceCollection& refList);

/**
 * Update sky coordinates in a collection of source objects
 *
 * @tparam SourceCollection  Type of sequence of sources, e.g. lsst::afw::table::SourceCatalog or
 *          std::vector<std::shared_ptr<lsst::afw::table::SourceRecord>>
 * @param[in] wcs  WCS to map from pixels to sky
 * @param[in,out] sourceList  Collection of sources. The schema must have two fields:
 *                  - "slot_Centroid": a field containing lsst::afw::geom::Point2D; this field is read
 *                  - "coord": a field containing an ICRS lsst::afw::SpherePoint; this field is written
 *
 * @throws lsst::pex::exceptions::NotFoundError if refList's schema does not have the required fields.
 */
template <typename SourceCollection>
void updateSourceCoords(geom::SkyWcs const& wcs, SourceCollection& sourceList);

}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // #ifndef LSST_AFW_TABLE_WCSUTILS_H
