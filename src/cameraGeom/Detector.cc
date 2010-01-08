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
    Eigen::Vector2d size;
    size << getAllPixels(isTrimmed).getWidth()*_pixelSize, getAllPixels(isTrimmed).getHeight()*_pixelSize;
    return afwGeom::Extent2D(size);
}

/**
 * Return the pixel position given an offset from the chip centre, in mm
 *
 * This base implementation assumes that all the pixels in the Detector are contiguous and of the same size
 */
afwGeom::Point2I cameraGeom::Detector::getIndexFromPosition(
        afwGeom::Point2D pos            ///< Offset from chip centre, mm
                                                           ) const
{
    Eigen::Vector2i pix;
    pix << _centerPixel[0] + pos[0]/_pixelSize, _centerPixel[1] + pos[1]/_pixelSize;
    return afwGeom::Point2I(pix);
}

/**
 * Return the offset from the chip centre, in mm, given a pixel position
 *
 * This base implementation assumes that all the pixels in the Detector are contiguous and of the same size
 */
afwGeom::Point2D cameraGeom::Detector::getPositionFromIndex(
        afwGeom::Point2I pix            ///< Pixel coordinates wrt bottom left of Detector
                                                           ) const
{
    Eigen::Vector2d pos;
    pos << (pix[0] - _centerPixel[0])*_pixelSize, (pix[1] - _centerPixel[1])*_pixelSize;
    return afwGeom::Point2D(pos);
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
