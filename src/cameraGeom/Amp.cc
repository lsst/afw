/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/Id.h"
#include "lsst/afw/cameraGeom/Amp.h"

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
        : _id(id), _isTrimmed(false), _allPixels(allPixels),
        _biasSec(biasSec), _dataSec(dataSec), _readoutCorner(readoutCorner),
        _eParams(eParams), _trimmedAllPixels()
{
    ;
}

/// Offset an Amp by the specified amount
void cameraGeom::Amp::shift(int dx,        ///< How much to offset in x (pixels)
                         int dy         ///< How much to offset in y (pixels)
                        ) {
    _allPixels.shift(dx, dy);
    _trimmedAllPixels.shift(dx, dy);
    _biasSec.shift(dx, dy);
    _dataSec.shift(dx, dy);
}
