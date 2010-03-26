#if !defined(LSST_AFW_CAMERAGEOM_DETECTOR_H)
#define LSST_AFW_CAMERAGEOM_DETECTOR_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Defect.h"
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

/**
 * Describe a detector (e.g. a CCD)
 */
class Detector {
public:
    typedef boost::shared_ptr<Detector> Ptr;
    typedef boost::shared_ptr<const Detector> ConstPtr;

    explicit Detector(
            Id id,                        ///< Detector's Id
            bool hasTrimmablePixels=false, ///< true iff Detector has pixels that can be trimmed (e.g. a CCD)
            double pixelSize=0.0           ///< Size of pixels, mm
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

    /// Are two Detectors identical, in the sense that they have the same name
    bool operator==(Detector const& rhs ///< Detector to compare too
                   ) const {
        return getId() == rhs.getId();
    }

    /// Is this less than rhs, based on comparing names
    bool operator<(Detector const& rhs  ///< Detector to compare too
                  ) const {
        return _id < rhs._id;
    }

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
    virtual lsst::afw::image::BBox& getAllPixels() {
        return (_hasTrimmablePixels && _isTrimmed) ? _trimmedAllPixels : _allPixels;
    }
    /// Return Detector's total footprint
    virtual lsst::afw::image::BBox getAllPixels() const {
        return getAllPixels(_isTrimmed);
    }
    /// Return Detector's total footprint
    virtual lsst::afw::image::BBox getAllPixels(
            bool isTrimmed ///< True iff the bias/overclock have been removed
    ) const {
        return (_hasTrimmablePixels && isTrimmed) ? _trimmedAllPixels : _allPixels;
    }
    //
    // Geometry of Detector --- i.e. mm not pixels
    //
    /// Set the central pixel
    void setCenterPixel(
            lsst::afw::geom::Point2I const& centerPixel ///< the pixel \e defined to be the detector's centre
                       ) { _centerPixel = centerPixel; }
    /// Return the central pixel
    lsst::afw::geom::Point2I getCenterPixel() const { return _centerPixel; }

    virtual void setOrientation(Orientation const& orientation);
    /// Return the Detector's Orientation
    Orientation const& getOrientation() const { return _orientation;}

    /// Set the Detector's center
    virtual void setCenter(lsst::afw::geom::Point2D const& center) { _center = center; }
    /// Return the Detector's center
    lsst::afw::geom::Point2D getCenter() const { return _center; }
    //
    // Translate between physical positions in mm to pixels
    //
    virtual lsst::afw::geom::Point2I getPixelFromPosition(lsst::afw::geom::Point2D const& pos) const;
    virtual lsst::afw::geom::Point2I getIndexFromPosition(lsst::afw::geom::Point2D const& pos) const;

    lsst::afw::geom::Point2D getPositionFromPixel(lsst::afw::geom::Point2I const& pix) const;
    lsst::afw::geom::Point2D getPositionFromPixel(
            lsst::afw::geom::Point2I const& pix,
            bool const isTrimmed
    ) const;
    virtual lsst::afw::geom::Point2D getPositionFromIndex(lsst::afw::geom::Point2I const& pix) const;
    virtual lsst::afw::geom::Point2D getPositionFromIndex(
            lsst::afw::geom::Point2I const& pix,
            bool const isTrimmed
    ) const;
    
    virtual void shift(int dx, int dy);
    //
    // Defects within this Detector
    //
    /// Set the Detector's Defect list
    virtual void setDefects(
            std::vector<boost::shared_ptr<lsst::afw::image::DefectBase> > const& defects
                ///< Defects in this detector
    ) {
        _defects = defects;
    }
    /// Get the Detector's Defect list
    std::vector<boost::shared_ptr<lsst::afw::image::DefectBase> > const& getDefects() const {
        return _defects;
    }
    std::vector<boost::shared_ptr<lsst::afw::image::DefectBase> >& getDefects() { return _defects; }
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
    lsst::afw::geom::Point2I _centerPixel;  // the pixel defined to be the centre of the Detector
    Orientation _orientation;           // orientation of this Detector
    lsst::afw::geom::Point2D _center;   // position of _centerPixel (mm)
    lsst::afw::geom::Extent2D _size;    // Size in mm of this Detector
    lsst::afw::image::BBox _trimmedAllPixels; // Bounding box of all the Detector's pixels after bias trimming

    std::vector<lsst::afw::image::DefectBase::Ptr> _defects; // Defects in this detector
};

namespace detail {
    template<typename T>
    struct sortPtr :
        public std::binary_function<typename T::Ptr &, typename T::Ptr const&, bool> {
        bool operator()(typename T::Ptr &lhs, typename T::Ptr const& rhs) const {
            return *lhs < *rhs;
        }
    };

    lsst::afw::image::BBox rotateBBoxBy90(
            lsst::afw::image::BBox const& bbox,
            lsst::afw::geom::Extent2I const& dimensions,
            int n90
                                         );
}
    
}}}

#endif
