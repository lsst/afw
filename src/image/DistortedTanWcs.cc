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
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/DistortedTanWcs.h"

namespace lsst {
namespace afw {
namespace image {

DistortedTanWcs::DistortedTanWcs(
    TanWcs const &tanWcs,
    geom::XYTransform const &pixelsToTanPixels
) : 
    TanWcs(tanWcs),
    _pixelsToTanPixelsPtr(pixelsToTanPixels.clone())
{
    if (tanWcs.hasDistortion()) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "tanWcs has distortion terms");
    }
}

PTR(Wcs) DistortedTanWcs::clone() const {
    return PTR(Wcs)(new DistortedTanWcs(*this));
}

bool DistortedTanWcs::operator==(Wcs const & rhs) const {
    throw LSST_EXCEPT(pex::exceptions::LogicError, "== is not implemented");
}

void DistortedTanWcs::flipImage(int flipLR, int flipTB, geom::Extent2I dimensions) const {
    throw LSST_EXCEPT(pex::exceptions::LogicError, "flipImage is not implemented");
}

void DistortedTanWcs::rotateImageBy90(int nQuarter, geom::Extent2I dimensions) const {
    throw LSST_EXCEPT(pex::exceptions::LogicError, "rotateImageBy90 is not implemented");
}

void DistortedTanWcs::shiftReferencePixel(double dx, double dy) {
    throw LSST_EXCEPT(pex::exceptions::LogicError, "shiftReferencePixel is not implemented");
}

geom::Point2D DistortedTanWcs::skyToPixelImpl(geom::Angle sky1, geom::Angle sky2) const {
    auto const tanPos = TanWcs::skyToPixelImpl(sky1, sky2);
    return _pixelsToTanPixelsPtr->reverseTransform(tanPos);
}

void DistortedTanWcs::pixelToSkyImpl(double pixel1, double pixel2, geom::Angle sky[2]) const {
    auto const pos = geom::Point2D(pixel1, pixel2);
    auto const tanPos = _pixelsToTanPixelsPtr->forwardTransform(pos);
    TanWcs::pixelToSkyImpl(tanPos[0], tanPos[1], sky);
}

}}} // namespace lsst::afw::image
