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
    typedef boost::shared_ptr<Detector> Ptr;
    typedef boost::shared_ptr<const Detector> ConstPtr;

    explicit Detector(
            Id id,                        //< Detector's Id
            bool hasTrimmablePixels=false, ///< true iff Detector has pixels that can be trimmed (e.g. a CCD)
            double pixelSize=0.0
                     ) :
        _id(id), _isTrimmed(false), _allPixels(),
        _hasTrimmablePixels(hasTrimmablePixels), _pixelSize(pixelSize)
    {
        if (_hasTrimmablePixels) {
            _trimmedAllPixels = _allPixels;
        }
    }
    virtual ~Detector() {}
    /// Return the Detector's Id
    Id getId() const { return _id; }

    /// Has the bias/overclock been removed?
    bool isTrimmed() const { return (_hasTrimmablePixels && _isTrimmed); }
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

    virtual lsst::afw::geom::Extent2D getSize() const;

    /// Return Detector's total footprint
    lsst::afw::image::BBox& getAllPixels() {
        return (_hasTrimmablePixels && _isTrimmed) ? _trimmedAllPixels : _allPixels;
    }
    lsst::afw::image::BBox getAllPixels() const {
        return getAllPixels(_isTrimmed);
    }
    lsst::afw::image::BBox getAllPixels(bool isTrimmed ///< True iff the bias/overclock have been removed
                                       ) const {
        return (_hasTrimmablePixels && isTrimmed) ? _trimmedAllPixels : _allPixels;
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

    virtual void shift(int dx, int dy);
protected:
    lsst::afw::image::BBox& getAllTrimmedPixels() {
        return _hasTrimmablePixels ? _trimmedAllPixels : _allPixels;
    }
private:
    Id _id;
    bool _isTrimmed;                    // Have all the bias/overclock regions been trimmed?
    lsst::afw::image::BBox _allPixels;  // Bounding box of all the Detector's pixels
    bool _hasTrimmablePixels;           // true iff Detector has pixels that can be trimmed (e.g. a CCD)
    double _pixelSize;                  // Size of a pixel in mm
    lsst::afw::geom::Point2I _centerPixel; // the pixel defined to be the centre of the Detector
    lsst::afw::geom::Extent2D _size;    // Size in mm of this Detector
    lsst::afw::image::BBox _trimmedAllPixels; // Bounding box of all the Detector's pixels after bias trimming
};

}}}

#endif
