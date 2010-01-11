/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/Id.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace cameraGeom = lsst::afw::cameraGeom;

/**
 * Return size in mm of this Detector
 */
afwGeom::Extent2D cameraGeom::Detector::getSize() const {
    bool const isTrimmed = true;

    return afwGeom::Extent2D::makeXY(getAllPixels(isTrimmed).getWidth()*_pixelSize,
                                     getAllPixels(isTrimmed).getHeight()*_pixelSize);
}

/**
 * Return the offset from the mosaic centre, in mm, given a pixel position
 * \sa getPositionFromIndex
 */
afwGeom::Point2D cameraGeom::Detector::getPositionFromPixel(
        afwGeom::Point2I const& pix,    ///< Pixel coordinates wrt bottom left of Detector
        bool const isTrimmed            ///< Is this detector trimmed?
                                                           ) const
{
    return getPositionFromIndex(pix - afwGeom::Extent2I(getCenterPixel()), isTrimmed);
}    

/**
 * Return the offset from the mosaic centre, in mm, given a pixel position
 * \sa getPositionFromIndex
 */
afwGeom::Point2D cameraGeom::Detector::getPositionFromPixel(
        afwGeom::Point2I const& pix     ///< Pixel coordinates wrt bottom left of Detector
                                                           ) const
{
    return getPositionFromPixel(pix, isTrimmed());
}    

/**
 * Return the pixel position given an offset from the chip centre, in mm
 *
 * This base implementation assumes that all the pixels in the Detector are contiguous and of the same size
 */
afwGeom::Point2I cameraGeom::Detector::getIndexFromPosition(
        afwGeom::Point2D const& pos     ///< Offset from chip centre, mm
                                                           ) const
{
    return afwGeom::Point2I::makeXY(_centerPixel[0] + pos[0]/_pixelSize, _centerPixel[1] + pos[1]/_pixelSize);
}

/**
 * Return the offset from the Detector centre, in mm, given a pixel position wrt Detector's centre
 * \sa getPositionFromPixel
 */
afwGeom::Point2D cameraGeom::Detector::getPositionFromIndex(
        afwGeom::Point2I const& pix     ///< Pixel coordinates wrt centre of Detector
                                     ) const
{
    return getPositionFromIndex(pix, isTrimmed());
}

/**
 * Return the offset from the Detector centre, in mm, given a pixel position wrt Detector's centre
 *
 * This base implementation assumes that all the pixels in the Detector are contiguous and of the same size
 * \sa getPositionFromPixel
 */
afwGeom::Point2D cameraGeom::Detector::getPositionFromIndex(
        afwGeom::Point2I const& pix,    ///< Pixel coordinates wrt centre of Detector
        bool const                      ///< Unused
                                                           ) const
{
    return afwGeom::Point2D::makeXY(_center[0] + pix[0]*_pixelSize, _center[1] + pix[1]*_pixelSize);
}    

/// Offset a Detector by the specified amount
void cameraGeom::Detector::shift(int dx, ///< How much to offset in x (pixels)
                                 int dy  ///< How much to offset in y (pixels)
                                ) {
    afwGeom::Extent2I offset(afwGeom::Point2I::makeXY(dx, dy));
    _centerPixel.shift(offset);
    
    _allPixels.shift(dx, dy);
    _trimmedAllPixels.shift(dx, dy);
}
