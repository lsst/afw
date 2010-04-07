#if !defined(LSST_AFW_CAMERAGEOM_DETECTOR_H)
#define LSST_AFW_CAMERAGEOM_DETECTOR_H

#include <string>
#include "boost/weak_ptr.hpp"
#include <boost/enable_shared_from_this.hpp>
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

#include "lsst/afw/image/Utils.h"
#include "lsst/afw/image/Defect.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
    
/**
 * Describe a detector (e.g. a CCD)
 */
    class Detector
#if !defined(SWIG)
        : public boost::enable_shared_from_this<Detector>
#endif
    {
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
        _parent = boost::weak_ptr<Detector>();
        
        if (_hasTrimmablePixels) {
            _trimmedAllPixels = _allPixels;
        }
    }
    virtual ~Detector() {}
    /// Return the Detector's Id
    Id getId() const { return _id; }

    /// Return the detector's parent in the hierarchy
    ///
    /// If the parent is unknown or has been deleted, and empty Ptr is returned
    void setParent(Ptr parent) {
        _parent = boost::weak_ptr<Detector>(parent);
    }

    /// Return the detector's parent in the hierarchy
    ///
    /// If the parent is unknown or has been deleted, and empty Ptr is returned
    Ptr getParent() const {
        return Ptr(_parent.lock());
    }

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

    virtual afwGeom::Extent2D getSize() const;

    /// Return Detector's total footprint
    virtual afwImage::BBox& getAllPixels() {
        return (_hasTrimmablePixels && _isTrimmed) ? _trimmedAllPixels : _allPixels;
    }
    /// Return Detector's total footprint
    virtual afwImage::BBox const& getAllPixels() const {
        return getAllPixels(_isTrimmed);
    }
    /// Return Detector's total footprint
    virtual afwImage::BBox const& getAllPixels(bool isTrimmed ///< Has the bias/overclock have been removed?
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
    //
    // Defects within this Detector
    //
    /// Set the Detector's Defect list
    virtual void setDefects(
            std::vector<boost::shared_ptr<afwImage::DefectBase> > const& defects ///< Defects in this detector
                           ) {
        _defects = defects;
    }
    /// Get the Detector's Defect list
    std::vector<boost::shared_ptr<afwImage::DefectBase> > const& getDefects() const { return _defects; }
    std::vector<boost::shared_ptr<afwImage::DefectBase> >& getDefects() { return _defects; }
protected:
    /// Return a shared pointer to this
    Ptr getThisPtr() {
        return shared_from_this();
    }

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
    boost::weak_ptr<Detector> _parent;  // Parent Detector in the hierarchy

    std::vector<afwImage::DefectBase::Ptr> _defects; // Defects in this detector
};

namespace detail {
    template<typename T>
    struct sortPtr :
        public std::binary_function<typename T::Ptr &, typename T::Ptr const&, bool> {
        bool operator()(typename T::Ptr &lhs, typename T::Ptr const& rhs) const {
            return *lhs < *rhs;
        }
    };

    afwImage::BBox rotateBBoxBy90(
            afwImage::BBox const& bbox,         ///< the BBox to rotate
            int n90,                            ///< number of 90-degree anti-clockwise turns to make
            afwGeom::Extent2I const& dimensions //< The size of the region wherein bbox dwells
                                 );
}
    
}}}

#endif
