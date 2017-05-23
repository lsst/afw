// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
#include "Eigen/Core"
#include "lsst/log/Log.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"

namespace except = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace image {

std::shared_ptr<Wcs> makeWcs(std::shared_ptr<daf::base::PropertySet> const& _metadata, bool stripMetadata) {
    //
    // _metadata is not const (it is probably meant to be), but we don't want to modify it.
    //
    auto metadata = _metadata;  // we'll make a copy and modify metadata if needs be
    auto modifyable = false;    // ... and set this variable to say that we did

    std::string ctype1, ctype2;
    if (metadata->exists("CTYPE1") && metadata->exists("CTYPE2")) {
        ctype1 = metadata->getAsString("CTYPE1");
        ctype2 = metadata->getAsString("CTYPE2");
    } else {
        LOGL_WARN("makeWcs", "Unable to construct valid WCS due to missing CTYPE1 or CTYPE2");
        return std::shared_ptr<Wcs>();
    }
    //
    // SCAMP used to use PVi_j keys with a CTYPE of TAN to specify a "TPV" projection
    // (cf. https://github.com/astropy/astropy/issues/299
    // and the discussion from Dave Berry in https://jira.lsstcorp.org/browse/DM-2883)
    //
    // Follow Dave's AST and switch TAN to TPV
    //
    if (ctype1.substr(5, 3) == "TAN" && (metadata->exists("PV1_5") || metadata->exists("PV2_1"))) {
        LOGL_INFO("makeWcs", "Interpreting %s/%s + PVi_j as TPV", ctype1.c_str(), ctype2.c_str());

        if (!modifyable) {
            metadata = _metadata->deepCopy();
            modifyable = true;
        }

        ctype1.replace(5, 3, "TPV");
        metadata->set<std::string>("CTYPE1", ctype1);

        ctype2.replace(5, 3, "TPV");
        metadata->set<std::string>("CTYPE2", ctype2);
    }

    std::shared_ptr<Wcs> wcs;  // we can't use make_shared as ctor is private
    if (ctype1.substr(5, 3) == "TAN") {
        wcs = std::shared_ptr<Wcs>(new TanWcs(metadata));
    } else if (ctype1.substr(5, 3) == "TPV") {  // unfortunately we don't support TPV
        if (!modifyable) {
            metadata = _metadata->deepCopy();
            modifyable = true;
        }

        LOGL_WARN("makeWcs", "Stripping PVi_j keys from projection %s/%s", ctype1.c_str(), ctype2.c_str());

        metadata->set<std::string>("CTYPE1", "RA---TAN");
        metadata->set<std::string>("CTYPE2", "DEC--TAN");
        metadata->set<bool>("TPV_WCS", true);
        // PV1_[1-4] are in principle legal, although Swarp reuses them as part of the TPV parameterisation.
        // It turns out that leaving PV1_[1-4] in the header breaks wcslib 4.14's ability to read
        // DECam headers (DM-3196) so we'll delete them all for now.
        //
        // John Swinbank points out the maximum value of j in TVi_j is 39;
        // http://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html
        for (int i = 0; i != 2; ++i) {
            for (int j = 1; j <= 39; ++j) {  // 39's the max in the TPV standard
                char pvName[8];
                sprintf(pvName, "PV%d_%d", i, j);
                if (metadata->exists(pvName)) {
                    metadata->remove(pvName);
                }
            }
        }

        wcs = std::shared_ptr<Wcs>(new TanWcs(metadata));
    } else {
        wcs = std::shared_ptr<Wcs>(new Wcs(metadata));
    }

    // If keywords LTV[1,2] are present, the image on disk is already a subimage, so
    // we should shift the wcs to allow for this.
    std::string key = "LTV1";
    if (metadata->exists(key)) {
        wcs->shiftReferencePixel(-metadata->getAsDouble(key), 0);
    }

    key = "LTV2";
    if (metadata->exists(key)) {
        wcs->shiftReferencePixel(0, -metadata->getAsDouble(key));
    }

    if (stripMetadata) {
        detail::stripWcsKeywords(_metadata, wcs);
    }

    return wcs;
}

std::shared_ptr<Wcs> makeWcs(coord::Coord const& crval, geom::Point2D const& crpix, double CD11, double CD12,
                             double CD21, double CD22) {
    Eigen::Matrix2d CD;
    CD << CD11, CD12, CD21, CD22;
    geom::Point2D crvalTmp;
    crvalTmp[0] = crval.toIcrs().getLongitude().asDegrees();
    crvalTmp[1] = crval.toIcrs().getLatitude().asDegrees();
    return std::shared_ptr<Wcs>(new TanWcs(crvalTmp, crpix, CD));
}
}
}
}  // end lsst::afw::image
