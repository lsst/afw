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
 
/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/Id.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/cameraGeom/Orientation.h"
#include "lsst/afw/cameraGeom/Distortion.h"
#include "lsst/afw/cameraGeom/FpPoint.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace pexEx = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace cameraGeom {


/************************************************************************************************************/
/// Test for equality of two Ids; ignore serial if < 0 and name if == ""
bool Id::operator==(Id const& rhs) const {
    if (_serial >= 0 && rhs._serial >= 0) {
        bool const serialEq = (_serial == rhs._serial);
        if (serialEq) {
            if (_name != "" && rhs._name != "") {
                return _name == rhs._name;
            }
        }
        
        return serialEq;
    } else {
        return _name == rhs._name;
    }
}

/// Test for ordering of two Ids; ignore serial if < 0 and name if == ""
bool Id::operator<(Id const& rhs) const {
    if (_serial >= 0 && rhs._serial >= 0) {
        if (_serial == rhs._serial) {
            if (_name != "" && rhs._name != "") {
                return _name < rhs._name;
            }
        }
        return _serial < rhs._serial;
    } else {
        return _name < rhs._name;
    }
}

/************************************************************************************************************/
/**
 * Return the Detector's footprint without applying any rotations that were used when inserting
 * it into its parent (e.g. Raft)
 */
afwGeom::BoxI
Detector::getAllPixelsNoRotation(bool isTrimmed ///< Has the bias/overclock have been removed?
                                 ) const
{
    afwGeom::BoxI allPixels = (_hasTrimmablePixels && isTrimmed) ? _trimmedAllPixels : _allPixels;

    int const n90 = _orientation.getNQuarter();
    if (n90 != 0) {
        allPixels = detail::rotateBBoxBy90(allPixels, -n90, getAllPixels(false).getDimensions());
    }

    return allPixels;
}

/************************************************************************************************************/

afwGeom::Point2D Detector::getCenterPixel() const {
    return afwGeom::Point2D(0.5*(getAllPixels(true).getWidth() - 1),
                            0.5*(getAllPixels(true).getHeight() - 1));
}

/************************************************************************************************************/
/**
 * Return size in mm of this Detector
 */
FpExtent Detector::getSize() const {
    bool const isTrimmed = true;

    return FpExtent(afwGeom::Extent2D(getAllPixels(isTrimmed).getWidth()*_pixelSize,
                                                  getAllPixels(isTrimmed).getHeight()*_pixelSize));
}

/**
 * Return the offset from the mosaic centre, in mm, given a pixel position
 */
FpPoint Detector::getPositionFromPixel(
    lsst::afw::geom::Point2D const& pix     ///< Pixel coordinates wrt bottom left of Detector
) const {
    double cosYaw = _orientation.getCosYaw();
    double sinYaw = _orientation.getSinYaw();

    afwGeom::Extent2D offset = afwGeom::Extent2D(cosYaw * pix.getX() - sinYaw * pix.getY(),
                                                 sinYaw * pix.getX() + cosYaw * pix.getY());
    offset -= afwGeom::Extent2D(getCenterPixel());
    offset *= getPixelSize();
    return _center + FpPoint(offset);
}


/**
 * Return the pixel position given an offset from the mosaic centre in mm
 */
afwGeom::Point2D Detector::getPixelFromPosition(
    FpPoint const &pos     ///< Offset from mosaic centre, mm
) const {
    double cosYaw = _orientation.getCosYaw();
    double sinYaw = _orientation.getSinYaw();

    afwGeom::Extent2D offset((pos - getCenter()).getPixels(getPixelSize()));
    offset += afwGeom::Extent2D(getCenterPixel());
    return afwGeom::Point2D(cosYaw * offset.getX() + sinYaw * offset.getY(),
                            -sinYaw * offset.getX() + cosYaw * offset.getY());
}


afwGeom::AffineTransform Detector::linearizePixelFromPosition(FpPoint const &q) const
{
    afwGeom::Point2D p = this->getPixelFromPosition(q);
    afwGeom::Point2D px = p + afwGeom::Extent2D(1,0);
    afwGeom::Point2D py = p + afwGeom::Extent2D(0,1);

    afwGeom::Point2D qx = this->getPositionFromPixel(px).getMm();
    afwGeom::Point2D qy = this->getPositionFromPixel(py).getMm();

    return afwGeom::makeAffineTransformFromTriple(q.getMm(), qx, qy, p, px, py);
}


afwGeom::AffineTransform Detector::linearizePositionFromPixel(afwGeom::Point2D const &p) const
{
    afwGeom::Point2D px = p + afwGeom::Extent2D(1,0);
    afwGeom::Point2D py = p + afwGeom::Extent2D(0,1);

    afwGeom::Point2D q = this->getPositionFromPixel(p).getMm();
    afwGeom::Point2D qx = this->getPositionFromPixel(px).getMm();
    afwGeom::Point2D qy = this->getPositionFromPixel(py).getMm();

    return afwGeom::makeAffineTransformFromTriple(p, px, py, q, qx, qy);
}


/// Offset a Detector by the specified amount
void Detector::shift(int dx, ///< How much to offset in x (pixels)
                     int dy  ///< How much to offset in y (pixels)
                     ) {
    afwGeom::Extent2I offset(dx, dy);
    
    _allPixels.shift(offset);
    _trimmedAllPixels.shift(offset);
}

/************************************************************************************************************/
/**
 * We're rotating an Image through an integral number of quarter turns,
 * modify this BBox accordingly
 *
 * If dimensions is provided interpret it as the size of an image, and the initial bbox as a bbox in
 * that image.  Then rotate about the center of the image
 *
 * If dimensions is 0, rotate the bbox about its LLC
 */
afwGeom::Box2I lsst::afw::cameraGeom::detail::rotateBBoxBy90(
        lsst::afw::geom::Box2I const& bbox,          ///< The BBox to rotate
        int n90,                                     ///< number of anti-clockwise 90degree turns
        lsst::afw::geom::Extent2I const& dimensions  ///< The size of the parent 
                                             )
{
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
    afwGeom::Point2I const centerPixel = afwGeom::Point2I(dimensions[0]/2, dimensions[1]/2);

    int x0, y0;                                          // minimum x/y
    int x1, y1;                                          // maximum x/y
    int xCorner[4], yCorner[4];                          // corners of Detector, wrt centerPixel

    int i = 0;
    xCorner[i] = bbox.getMinX() - centerPixel[0];
    yCorner[i] = bbox.getMinY() - centerPixel[1];
    ++i;

    xCorner[i] = bbox.getMaxX() - centerPixel[0];
    yCorner[i] = bbox.getMinY() - centerPixel[1];
    ++i;

    xCorner[i] = bbox.getMaxX() - centerPixel[0];
    yCorner[i] = bbox.getMaxY() - centerPixel[1];
    ++i;

    xCorner[i] = bbox.getMinX() - centerPixel[0];
    yCorner[i] = bbox.getMaxY() - centerPixel[1];
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
        
    afwGeom::Point2I LLC(centerPixel[0] + x0, centerPixel[1] + y0);
    afwGeom::Point2I URC(centerPixel[0] + x1, centerPixel[1] + y1);
        
    afwGeom::Box2I newBbox(LLC, URC);
        
    //int const dxy0 = (dimensions[1]/2 - dimensions[0]/2); // how far the origin moved
    
    int const dxy0 = centerPixel[0] - centerPixel[1];
    if (n90%2 == 1 && dxy0 != 0) {
        newBbox.shift(geom::Extent2I(-dxy0, dxy0));
    }
        
    return newBbox;
}

///
/// Set the Detector's Orientation
///
void Detector::setOrientation(
       Orientation const& orientation // the detector's new Orientation
                              )
{
    int const n90 = orientation.getNQuarter() - _orientation.getNQuarter();
    _orientation = orientation;
    //
    // Now update the private members
    //
    _allPixels = detail::rotateBBoxBy90(
        _allPixels, n90, getAllPixels(false).getDimensions()
    );
    _trimmedAllPixels = detail::rotateBBoxBy90(
        _trimmedAllPixels, n90, getAllPixels(true).getDimensions()
    );
    if (n90 == 1 || n90 == 3) {
        _size = afwGeom::Extent2D(_size[1], _size[0]);
    }
}


void Detector::setDistortion(CONST_PTR(Distortion) distortion) {
    _distortion = distortion;
}

CONST_PTR(Distortion) Detector::getDistortion() const {
    
    // if we have a distortion ... return it
    if (_distortion) {
        return _distortion;
        
        // otherwise, return our parent's ... no parent? return Null 
    } else {
        CONST_PTR(Detector) parent = this->getParent();
        if (parent) {
            return parent->getDistortion();
        } else {
            return CONST_PTR(Distortion)(); //new Distortion());
        }
    }
    
    return CONST_PTR(Distortion)(); //new Distortion());
}



// -----------------------------------------------------------------------------------------------------------
//
// DetectorXYTransform


DetectorXYTransform::DetectorXYTransform(CONST_PTR(XYTransform) fpTransform, 
                                         CONST_PTR(Detector) detector)
    : XYTransform(false), _fpTransform(fpTransform), _detector(detector)
{
    if (!fpTransform->inFpCoordinateSystem()) {
        throw LSST_EXCEPT(pexEx::InvalidParameterException, 
                          "DetectorXYTransform base must be in FP coordinate system");
    }
}

PTR(geom::XYTransform) DetectorXYTransform::clone() const
{
    // note: there is no detector->clone()
    return boost::make_shared<DetectorXYTransform> (_fpTransform->clone(), _detector);
}

PTR(geom::XYTransform) DetectorXYTransform::invert() const
{
    return boost::make_shared<DetectorXYTransform> (_fpTransform->invert(), _detector);
}

geom::Point2D DetectorXYTransform::forwardTransform(Point2D const &p) const
{
    Point2D q;
    q = _detector->getPositionFromPixel(p).getMm();
    q = _fpTransform->forwardTransform(q);
    q = _detector->getPixelFromPosition(FpPoint(q));
    return q;
}

geom::Point2D DetectorXYTransform::reverseTransform(Point2D const &p) const
{
    Point2D q;
    q = _detector->getPositionFromPixel(p).getMm();
    q = _fpTransform->reverseTransform(q);
    q = _detector->getPixelFromPosition(FpPoint(q));
    return q;
}

geom::AffineTransform DetectorXYTransform::linearizeForwardTransform(Point2D const &p) const
{
    AffineTransform a;
    a = _detector->linearizePositionFromPixel(p);
    a = _fpTransform->linearizeForwardTransform(a(p)) * a;
    a = _detector->linearizePixelFromPosition(FpPoint(a(p))) * a;
    return a;
}

geom::AffineTransform DetectorXYTransform::linearizeReverseTransform(Point2D const &p) const
{
    AffineTransform a;
    a = _detector->linearizePositionFromPixel(p);
    a = _fpTransform->linearizeReverseTransform(a(p)) * a;
    a = _detector->linearizePixelFromPosition(FpPoint(a(p))) * a;
    return a;    
}

}}}
