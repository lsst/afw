/*
 * LSST Data Management System
 * Copyright 2017 LSST Corporation.
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

#include <iostream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <limits>

#include "boost/format.hpp"

#include "wcslib/wcs.h"
#include "wcslib/wcsfix.h"
#include "wcslib/wcshdr.h"

#include "lsst/daf/base.h"
#include "lsst/afw/geom/detail/wcsUtils.h"

namespace lsst {
namespace afw {
namespace geom {
namespace detail {

/**
 * @internal Define a trivial WCS that maps the lower left corner (LLC) pixel of an image to a given value
 */
std::shared_ptr<daf::base::PropertyList> createTrivialWcsAsPropertySet(
        std::string const& wcsName,  ///< @internal Name of desired WCS
        int const x0,                ///< @internal Column coordinate of LLC pixel
        int const y0                 ///< @internal Row coordinate of LLC pixel
) {
    std::shared_ptr<daf::base::PropertyList> wcsMetaData(new daf::base::PropertyList);

    wcsMetaData->set("CRVAL1" + wcsName, x0, "Column pixel of Reference Pixel");
    wcsMetaData->set("CRVAL2" + wcsName, y0, "Row pixel of Reference Pixel");
    wcsMetaData->set("CRPIX1" + wcsName, 1, "Column Pixel Coordinate of Reference");
    wcsMetaData->set("CRPIX2" + wcsName, 1, "Row Pixel Coordinate of Reference");
    wcsMetaData->set("CTYPE1" + wcsName, "LINEAR", "Type of projection");
    wcsMetaData->set("CTYPE2" + wcsName, "LINEAR", "Type of projection");
    wcsMetaData->set("CUNIT1" + wcsName, "PIXEL", "Column unit");
    wcsMetaData->set("CUNIT2" + wcsName, "PIXEL", "Row unit");

    return wcsMetaData;
}

/**
 * @internal Return a Point2I(x0, y0) given a PropertySet containing a suitable WCS (e.g. "A")
 *
 * The WCS must have CRPIX[12] == 1 and CRVAL[12] must be present.  If this is true, the WCS
 * cards are removed from the metadata
 *
 * @param wcsName the WCS to search (E.g. "A")
 * @param metadata the metadata, maybe containing the WCS
 */
geom::Point2I getImageXY0FromMetadata(std::string const& wcsName, daf::base::PropertySet* metadata) {
    int x0 = 0;  // Our value of X0
    int y0 = 0;  // Our value of Y0

    try {
        //
        // Only use WCS if CRPIX[12] == 1 and CRVAL[12] is present
        //
        if (metadata->getAsDouble("CRPIX1" + wcsName) == 1 &&
            metadata->getAsDouble("CRPIX2" + wcsName) == 1) {
            x0 = metadata->getAsInt("CRVAL1" + wcsName);
            y0 = metadata->getAsInt("CRVAL2" + wcsName);
            //
            // OK, we've got it.  Remove it from the header
            //
            metadata->remove("CRVAL1" + wcsName);
            metadata->remove("CRVAL2" + wcsName);
            metadata->remove("CRPIX1" + wcsName);
            metadata->remove("CRPIX2" + wcsName);
            metadata->remove("CTYPE1" + wcsName);
            metadata->remove("CTYPE1" + wcsName);
            metadata->remove("CUNIT1" + wcsName);
            metadata->remove("CUNIT2" + wcsName);
        }
    } catch (pex::exceptions::NotFoundError&) {
        ;  // OK, not present
    }

    return geom::Point2I(x0, y0);
}

}  // namespace detail
}  // namespace geom
}  // namespace afw
}  // namespace lsst
