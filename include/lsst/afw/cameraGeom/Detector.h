#if !defined(LSST_AFW_CAMERAGEOM_DETECTOR_H)
#define LSST_AFW_CAMERAGEOM_DETECTOR_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Id.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Describe a detector (e.g. a CCD)
 */
class Detector {
public:
    Detector(Id id, double pixelSize=0.0) : _id(id), _isTrimmed(false), _allPixels(), _pixelSize(pixelSize) {
        _trimmedAllPixels = _allPixels;
    }
    virtual ~Detector() {}
    /// Return the Detector's Id
    Id getId() const { return _id; }

    /// Has the bias/overclock been removed?
    bool isTrimmed() const { return _isTrimmed; }
    /// Set the trimmed status of this Detector
    virtual void setTrimmed(bool isTrimmed      ///< True iff the bias/overclock have been removed
                           ) {
        _isTrimmed = isTrimmed;
    }
    
    /// Set the pixel size in mm
    void setPixelSize(double pixelSize  ///< Size of a pixel, mm
                     ) { _pixelSize = pixelSize; }
    /// Return the pixel size, mm/pixel
    double getPixelSize() const { return _pixelSize; }

    /// Return size in mm of this Detector
    lsst::afw::geom::Extent2D getSize() const {
        return getSize(_isTrimmed);
    }
    
    lsst::afw::geom::Extent2D getSize(bool isTrimmed ///< True iff the bias/overclock have been removed
                                     ) const {
        Eigen::Vector2d size;
        size << getAllPixels(isTrimmed).getWidth()*_pixelSize, getAllPixels(isTrimmed).getHeight()*_pixelSize;
        return lsst::afw::geom::Extent2D(size);
    }

    /// Return Detector's total footprint
    lsst::afw::image::BBox& getAllPixels() {
        return _isTrimmed ? _trimmedAllPixels : _allPixels;
    }
    lsst::afw::image::BBox getAllPixels() const {
        return getAllPixels(_isTrimmed);
    }
    lsst::afw::image::BBox getAllPixels(bool isTrimmed ///< True iff the bias/overclock have been removed
                                       ) const {
        return isTrimmed ? _trimmedAllPixels : _allPixels;
    }
    //
    // Geometry of Detector --- i.e. mm not pixels
    //
    /// Set the central pixel
    void setCenterPixel(lsst::afw::geom::Point2I center ///< the pixel \e defined to be the detector's centre
                       ) { _centerPixel = center; }
    /// Return the central pixel
    lsst::afw::geom::Point2I getCenterPixel() const { return _centerPixel; }
    //
    // Translate between physical positions in mm to pixels
    //
    virtual lsst::afw::geom::Point2I getIndexFromPosition(lsst::afw::geom::Point2D pos) const;
    virtual lsst::afw::geom::Point2D getPositionFromIndex(lsst::afw::geom::Point2I pos) const;
protected:
    lsst::afw::image::BBox& getAllTrimmedPixels() {
        return _trimmedAllPixels;
    }
private:
    Id _id;
    bool _isTrimmed;                    // Have all the bias/overclock regions been trimmed?
    lsst::afw::image::BBox _allPixels;  // Bounding box of all the Detector's pixels
    double _pixelSize;                  // Size of a pixel in mm
    lsst::afw::geom::Point2I _centerPixel; // the pixel defined to be the centre of the Detector
    lsst::afw::geom::Extent2D _size;    // Size in mm of this Detector
    lsst::afw::image::BBox _trimmedAllPixels; // Bounding box of all the Detector's pixels after bias trimming
};

}}}

#endif
