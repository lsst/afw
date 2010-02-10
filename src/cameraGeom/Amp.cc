/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/Id.h"
#include "lsst/afw/cameraGeom/Amp.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace cameraGeom = lsst::afw::cameraGeom;

cameraGeom::ElectronicParams::ElectronicParams(
        float gain,                     ///< Amplifier's gain
        float readNoise,                ///< Amplifier's read noise (DN)
        float saturationLevel           ///< Amplifier's saturation level. N.b. float in case we scale data
                                           )
    : _gain(gain), _readNoise(readNoise), _saturationLevel(saturationLevel)
{}

/************************************************************************************************************/

cameraGeom::Amp::Amp(
        cameraGeom::Id id,                            ///< The amplifier's ID
        afwImage::BBox const& allPixels,           ///< Bounding box of the pixels read off this amplifier
        afwImage::BBox const& biasSec,             ///< Bounding box of amplifier's bias section
        afwImage::BBox const& dataSec,             ///< Bounding box of amplifier's data section
        cameraGeom::Amp::ReadoutCorner readoutCorner, ///< location of first pixel read
        ElectronicParams::Ptr eParams              ///< electronic properties of Amp
                 )
    : Detector(id, true),
    _biasSec(biasSec), _dataSec(dataSec), _readoutCorner(readoutCorner), _eParams(eParams)
{
    if (biasSec.getWidth() > 0 && biasSec.getHeight() > 0 &&
        (!allPixels.contains(biasSec.getLLC()) || !allPixels.contains(biasSec.getURC()))) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("%||'s bias section doesn't fit in allPixels") % id).str());
    }
    if (dataSec.getWidth() > 0 && dataSec.getHeight() > 0 &&
        (!allPixels.contains(dataSec.getLLC()) || !allPixels.contains(dataSec.getURC()))) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("%||'s data section doesn't fit in allPixels") % id).str());
        
    }

    getAllPixels() = allPixels;
    
    setTrimmedGeom();
}
/**
 * Set the geometry of the Amp after trimming the overclock/extended regions
 *
 * Note that the trimmed BBoxes are relative to a trimmed detector
 */
void cameraGeom::Amp::setTrimmedGeom() {
    //
    // Figure out which Amp we are
    //
    int const iX = getAllPixels().getX0()/getAllPixels().getWidth();
    int const iY = getAllPixels().getY0()/getAllPixels().getHeight();
    
    int const dataHeight = _dataSec.getHeight();
    int const dataWidth = _dataSec.getWidth();

    _trimmedDataSec = afwImage::BBox(afwImage::PointI(iX*dataWidth, iY*dataHeight), dataWidth, dataHeight);
    getAllTrimmedPixels() = _trimmedDataSec;
}

/// Offset an Amp by the specified amount
void cameraGeom::Amp::shift(int dx,        ///< How much to offset in x (pixels)
                            int dy         ///< How much to offset in y (pixels)
                        ) {
    getAllPixels().shift(dx, dy);
    _biasSec.shift(dx, dy);
    _dataSec.shift(dx, dy);
    getAllTrimmedPixels().shift(dx, dy);
    _trimmedDataSec.shift(dx, dy);
}

/// Rotate an Amp by some number of 90degree anticlockwise turns about centerPixel
void cameraGeom::Amp::rotateBy90(
        afwGeom::Extent2I const& dimensions, ///< Size of CCD
        int n90                              ///< How many 90-degree anti-clockwise turns should I apply?
                                ) {
    while (n90 < 0) {
        n90 += 4;
    }
    n90 %= 4;                           // normalize

    if (n90 == 0) {                     // nothing to do.  Good.
        return;
    }
    //
    // Rotate the amps to the right orientation
    //
    getAllPixels() = cameraGeom::detail::rotateBBoxBy90(getAllPixels(), dimensions, n90);
    _biasSec = cameraGeom::detail::rotateBBoxBy90(_biasSec, dimensions, n90);
    _dataSec = cameraGeom::detail::rotateBBoxBy90(_dataSec, dimensions, n90);

    setTrimmedGeom();
    //
    // Fix the readout corner
    //
    _readoutCorner = static_cast<ReadoutCorner>((_readoutCorner + n90)%4);
}
