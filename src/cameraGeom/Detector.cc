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

    return afwGeom::makeExtentD(getAllPixels(isTrimmed).getWidth()*_pixelSize,
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
 * Return the offset in pixels from the detector centre, given an offset from the detector centre in mm
 *
 * This base implementation assumes that all the pixels in the Detector are contiguous and of the same size
 */
afwGeom::Point2I cameraGeom::Detector::getIndexFromPosition(
        afwGeom::Point2D const& pos     ///< Offset from chip centre, mm
                                                           ) const
{
    return afwGeom::makePointI(pos[0]/_pixelSize, pos[1]/_pixelSize);
}

/**
 * Return the pixel position given an offset from the mosaic centre in mm
 * \sa getIndexFromPosition
 */
afwGeom::Point2I cameraGeom::Detector::getPixelFromPosition(
        afwGeom::Point2D const& pos     ///< Offset from mosaic centre, mm
                                                                 ) const
{
    return afwGeom::Extent2I(getCenterPixel()) + getIndexFromPosition(pos - afwGeom::Extent2D(getCenter()));
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
    return afwGeom::makePointD(_center[0] + pix[0]*_pixelSize, _center[1] + pix[1]*_pixelSize);
}    

/// Offset a Detector by the specified amount
void cameraGeom::Detector::shift(int dx, ///< How much to offset in x (pixels)
                                 int dy  ///< How much to offset in y (pixels)
                                ) {
    afwGeom::Extent2I offset(afwGeom::makePointI(dx, dy));
    _centerPixel.shift(offset);
    
    _allPixels.shift(dx, dy);
    _trimmedAllPixels.shift(dx, dy);
}

/************************************************************************************************************/
/**
 * We're rotating an Image through an integral number of quarter turns,
 * modify this BBox accordingly
 */
afwImage::BBox cameraGeom::detail::rotateBBoxBy90(
        afwImage::BBox const& bbox,          ///< The BBox to rotate
        afwGeom::Extent2I const& dimensions, ///< The size of the parent 
        int n90                              ///< number of anti-clockwise 90degree turns
                                             )
{
    afwGeom::Point2I const centerPixel = afwGeom::makePointI(dimensions[0]/2, dimensions[1]/2);

    while (n90 < 0) {
        n90 += 4;
    }
    n90 %= 4;
    
    int s, c;                           // sin/cos of the rotation angle
    switch (n90) {
      case 0:
        s = 0; c = 1;
        break;
      case 1:
        s = 1; c = 0;
        break;
      case 2:
        s = 0; c = -1;
        break;
      case 3:
        s = -1; c = 0;
        break;
      default:
        c = s = 0;                      // make compiler happy
        assert(n90 >= 0 && n90 <= 3);   // we said "%= 4"
    }
    //
    // To work
    //
    int x0, y0;                                          // minimum x/y
    int x1, y1;                                          // maximum x/y
    int xCorner[4], yCorner[4];                          // corners of Detector, wrt centerPixel

    int i = 0;
    xCorner[i] = bbox.getX0() - centerPixel[0];
    yCorner[i] = bbox.getY0() - centerPixel[1];
    ++i;

    xCorner[i] = bbox.getX1() - centerPixel[0];
    yCorner[i] = bbox.getY0() - centerPixel[1];
    ++i;

    xCorner[i] = bbox.getX1() - centerPixel[0];
    yCorner[i] = bbox.getY1() - centerPixel[1];
    ++i;

    xCorner[i] = bbox.getX0() - centerPixel[0];
    yCorner[i] = bbox.getY1() - centerPixel[1];
    ++i;
    //
    // Now see which is the smallest/largest
    i = 0;
    x0 = x1 = c*xCorner[i] - s*yCorner[i];
    y0 = y1 = s*xCorner[i] + c*yCorner[i];
    ++i;

    for (; i != 4; ++i) {
        int x = c*xCorner[i] - s*yCorner[i];
        int y = s*xCorner[i] + c*yCorner[i];

        if (x < x0) {
            x0 = x;
        }
        if (x > x1) {
            x1 = x;
        }
        if (y < y0) {
            y0 = y;
        }
        if (y > y1) {
            y1 = y;
        }
    }
    //
    // Fiddle things a little if the detector has an even number of pixels so that square BBoxes
    // will map into themselves
    //
    if(n90 == 1) {
        if (dimensions[0]%2 == 0) {
            x0--; x1--;
        }
    } else if (n90 == 2) {
        if (dimensions[0]%2 == 0) {
            x0--; x1--;
        }
        if (dimensions[1]%2 == 0) {
            y0--; y1--;
        }
    } else if(n90 == 3) {
        if (dimensions[1]%2 == 0) {
            y0--; y1--;
        }
    }

    afwImage::PointI LLC(centerPixel[0] + x0, centerPixel[1] + y0);
    afwImage::PointI URC(centerPixel[0] + x1, centerPixel[1] + y1);

    afwImage::BBox newBbox(LLC, URC);

    int const dxy0 = (dimensions[1] - dimensions[0])/2; // how far the origin moved
    if (n90%2 == 1 && dxy0 != 0) {
        newBbox.shift(dxy0, -dxy0);
    }

    return newBbox;
}

///
/// Set the Detector's Orientation
///
void cameraGeom::Detector::setOrientation(
        cameraGeom::Orientation const& orientation // the detector's new Orientation
                                         )
{
    int const n90 = orientation.getNQuarter() - _orientation.getNQuarter();
    _orientation = orientation;
    //
    // Now update the private members
    //
    _allPixels =
        cameraGeom::detail::rotateBBoxBy90(_allPixels,
                                           afwGeom::makeExtentI(getAllPixels(false).getWidth(),
                                                                     getAllPixels(false).getHeight()), n90);
    _trimmedAllPixels =
        cameraGeom::detail::rotateBBoxBy90(_trimmedAllPixels,
                                           afwGeom::makeExtentI(getAllPixels(true).getWidth(),
                                                                     getAllPixels(true).getHeight()), n90);
        
    if (n90 == 1 || n90 == 3) {
        _size = afwGeom::makeExtentD(_size[1], _size[0]);
    }
}
