// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
