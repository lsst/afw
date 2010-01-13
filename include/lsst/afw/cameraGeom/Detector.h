#if !defined(LSST_AFW_CAMERAGEOM_DETECTOR_H)
#define LSST_AFW_CAMERAGEOM_DETECTOR_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Id.h"
#include "lsst/afw/cameraGeom/Orientation.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
    
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

    virtual afwGeom::Extent2D getSize() const;

    /// Return Detector's total footprint
    virtual afwImage::BBox& getAllPixels() {
        return (_hasTrimmablePixels && _isTrimmed) ? _trimmedAllPixels : _allPixels;
    }
    virtual afwImage::BBox getAllPixels() const {
        return getAllPixels(_isTrimmed);
    }
    virtual afwImage::BBox getAllPixels(bool isTrimmed ///< True iff the bias/overclock have been removed
                                       ) const {
        return (_hasTrimmablePixels && isTrimmed) ? _trimmedAllPixels : _allPixels;
    }
    //
    // Geometry of Detector --- i.e. mm not pixels
    //
    /// Set the central pixel
    void setCenterPixel(
            afwGeom::Point2I const& centerPixel ///< the pixel \e defined to be the detector's centre
                       ) { _centerPixel = centerPixel; }
    /// Return the central pixel
    afwGeom::Point2I getCenterPixel() const { return _centerPixel; }

    virtual void setOrientation(Orientation const& orientation);
    /// Return the Detector's Orientation
    Orientation const& getOrientation() const { return _orientation;}

    /// Set the Detector's center
    virtual void setCenter(afwGeom::Point2D const& center) { _center = center; }
    /// Return the Detector's center
    afwGeom::Point2D getCenter() const { return _center; }
    //
    // Translate between physical positions in mm to pixels
    //
    virtual afwGeom::Point2I getPixelFromPosition(afwGeom::Point2D const& pos) const;
    virtual afwGeom::Point2I getIndexFromPosition(afwGeom::Point2D const& pos) const;

    afwGeom::Point2D getPositionFromPixel(afwGeom::Point2I const& pix) const;
    afwGeom::Point2D getPositionFromPixel(afwGeom::Point2I const& pix, bool const isTrimmed) const;
    virtual afwGeom::Point2D getPositionFromIndex(afwGeom::Point2I const& pix) const;
    virtual afwGeom::Point2D getPositionFromIndex(afwGeom::Point2I const& pix, bool const isTrimmed) const;
    
    virtual void shift(int dx, int dy);
protected:
    afwImage::BBox& getAllTrimmedPixels() {
        return _hasTrimmablePixels ? _trimmedAllPixels : _allPixels;
    }
private:
    Id _id;
    bool _isTrimmed;                    // Have all the bias/overclock regions been trimmed?
    afwImage::BBox _allPixels;          // Bounding box of all the Detector's pixels
    bool _hasTrimmablePixels;           // true iff Detector has pixels that can be trimmed (e.g. a CCD)
    double _pixelSize;                  // Size of a pixel in mm
    afwGeom::Point2I _centerPixel;      // the pixel defined to be the centre of the Detector
    Orientation _orientation;           // orientation of this Detector
    afwGeom::Point2D _center;           // position of _centerPixel (mm)
    afwGeom::Extent2D _size;            // Size in mm of this Detector
    afwImage::BBox _trimmedAllPixels;   // Bounding box of all the Detector's pixels after bias trimming
};

namespace detail {
    afwImage::BBox rotateBBoxBy90(afwImage::BBox const& bbox, afwGeom::Extent2I const& dimensions, int n90);
}
    
}}}

#endif
