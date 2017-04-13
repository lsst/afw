// -*- lsst-c++ -*-
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

#ifndef LSST_AFW_GEOM_FRAMESETUTILS_H
#define LSST_AFW_GEOM_FRAMESETUTILS_H

#include <memory>
#include <vector>

#include "astshim.h"
#include "ndarray.h"

#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/daf/base/PropertyList.h"

namespace lsst {
namespace afw {
namespace geom {
namespace detail {

/**
Get the specified frame of a FrameSet as a SkyFrame

@param[in] frameSet  FrameSet from which to get the SkyFrame
@param[in] index  Index of SkyFrame.
        This value should lie in the range 1 to the number of frames in the FrameSet
        (as given by getNframe). A value of FrameSet::Base or FrameSet::CURRENT
        may be given to specify the base Frame or the current Frame, respectively.
@param[in] copy  If true return a deep copy of the SkyFrame, else a shallow copy
*/
std::shared_ptr<ast::SkyFrame> getSkyFrame(ast::FrameSet const& frameSet, int index, bool copy);

/**
Make FITS metadata for a pure tangent WCS

@param[in] crpix  Center of projection on the CCD using the LSST convention:
                        0, 0 is the lower left pixel of the image
@param[in] crval  Center of projection on the sky
@param[in] cdMatrix  CD matrix where element (i-1, j-1) corresponds to FITS keyword CDi_j
                        and i, j have range [1, 2]
*/
std::shared_ptr<daf::base::PropertyList> makeTanWcsMetadata(Point2D const& crpix, SpherePoint const& crval,
                                                            Eigen::Matrix2d const& cdMatrix);

/**
Read a FITS convention WCS FrameSet from FITS metadata

The resulting FrameSet may be any kind of WCS supported by FITS;
if it is a celestial WCS then 1,1 will be the lower left corner of the image
(the FITS convention, not the LSST convention).

@todo Add the ability to purge read fits cards.

@param[in,out] metadata  FITS header cards
@param[in] strip  If true strip items from `metadata` that are used to create the WCS
*/
std::shared_ptr<ast::FrameSet> readFitsWcs(daf::base::PropertyList& metadata, bool strip=true);

/**
Read an LSST sky WCS FrameSet from a FITS header: 0,0 is the lower left corner of the parent image

Read standard FITS WCS header cards plus optional LSST-specific keywords "LTV1" and "LTV2",
which, if found, specify the position of the subimage in the parent image.

Set the output SkyFrame's SkyRef to CRVAL and SkyRefIs="Ignore" so the SkyRef is ignored
instead of being treated as an offset.

@warning the general WCS generated by LSST software cannot be exactly represented using
standard WCS FITS cards, and so this function cannot read those. This function is intended
for two purposes:
- Read the standard FITS WCS found in raw data and other images from non-LSST observatories
    and convert it to the LSST pixel convention.
- Read the approximate FITS WCS that LSST writes to FITS images (for use by non-LSST code).

The frames of the returned WCS will be as follows:
- base: an ast::Frame with domain "PIXELS0": pixels with 0,0 the lower left corner of the parent image
- current: an ast::SkyFrame: ICRS RA, Dec

@todo Add the ability to purge read fits cards.

@param[in,out] metadata  FITS header cards
@param[in] strip  If true strip items from `metadata` that are used to create the WCS,
    excluding `LTV1` and `LTV2`, which are always retained.
*/
std::shared_ptr<ast::FrameSet> readLsstSkyWcs(daf::base::PropertyList& metadata, bool strip=false);

}  // namespace detail
}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif
