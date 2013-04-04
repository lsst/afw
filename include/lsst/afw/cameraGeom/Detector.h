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
 
#if !defined(LSST_AFW_CAMERAGEOM_DETECTOR_H)
#define LSST_AFW_CAMERAGEOM_DETECTOR_H

#include <string>
#include "boost/weak_ptr.hpp"
#include <boost/enable_shared_from_this.hpp>
#include "lsst/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Defect.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Id.h"
#include "lsst/afw/cameraGeom/Orientation.h"

#include "lsst/afw/cameraGeom/FpPoint.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {
 
class Distortion;
    
/**
 * Describe a detector (e.g. a CCD)
 */
    class Detector
        : public lsst::daf::base::Citizen
#if !defined(SWIG)
        , public boost::enable_shared_from_this<Detector>
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
        lsst::daf::base::Citizen(typeid(this)),
        _id(id), _isTrimmed(false), _allPixels(),
        _hasTrimmablePixels(hasTrimmablePixels), _pixelSize(pixelSize), _center(),
        _distortion(PTR(Distortion)()) // init as a null pointer
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
    /// If the parent is unknown or has been deleted, an empty Ptr is returned
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

    virtual FpExtent getSize() const;

    /// Return Detector's total footprint
    virtual lsst::afw::geom::Box2I& getAllPixels() {
        return (_hasTrimmablePixels && _isTrimmed) ? _trimmedAllPixels : _allPixels;
    }
    /// Return Detector's total footprint
    virtual lsst::afw::geom::Box2I const& getAllPixels() const {
        return getAllPixels(_isTrimmed);
    }
    /// Return Detector's total footprint
    virtual lsst::afw::geom::Box2I const& getAllPixels(bool isTrimmed ///< Has the bias/overclock have been removed?
                                              ) const {
        return (_hasTrimmablePixels && isTrimmed) ? _trimmedAllPixels : _allPixels;
    }
    /// Return the Detector's footprint without applying any rotations that were used when inserting
    /// it into its parent (e.g. Raft)
    virtual lsst::afw::geom::BoxI getAllPixelsNoRotation(bool isTrimmed=true) const;

    /// Get the center pixel of the detector
    geom::Point2D getCenterPixel() const;

    //
    // Geometry of Detector --- i.e. mm not pixels
    //
    virtual void setOrientation(Orientation const& orientation);
    /// Return the Detector's Orientation
    Orientation const& getOrientation() const { return _orientation;}

    /// Set the Detector's center
    virtual void setCenter(FpPoint const& center) { _center = center; }

    /// Set the Detector's center relative to the old center.
    virtual void shiftCenter(FpExtent const & offset) { _center = FpPoint(_center.getMm() + offset.getMm()); }

    /// Return the Detector's center
    FpPoint getCenter() const { return _center; }

    /// Translate between physical positions in mm to pixels
    virtual lsst::afw::geom::Point2D getPixelFromPosition(FpPoint const& pos) const;
    virtual FpPoint getPositionFromPixel(lsst::afw::geom::Point2D const& pix) const;

    /// Local linearizations of the preceding coordinate-changing routines
    virtual lsst::afw::geom::AffineTransform linearizePixelFromPosition(FpPoint const &pos) const;
    virtual lsst::afw::geom::AffineTransform linearizePositionFromPixel(lsst::afw::geom::Point2D const &pix) const;
    
    virtual void shift(int dx, int dy);
    //
    // Defects within this Detector
    //
    /// Set the Detector's Defect list
    virtual void setDefects(
            std::vector<boost::shared_ptr<lsst::afw::image::DefectBase> > const& defects ///< Defects in this detector
                           ) {
        _defects = defects;
    }
    /// Get the Detector's Defect list
    std::vector<boost::shared_ptr<lsst::afw::image::DefectBase> > const& getDefects() const { return _defects; }
    std::vector<boost::shared_ptr<lsst::afw::image::DefectBase> >& getDefects() { return _defects; }

    void setDistortion(CONST_PTR(Distortion) distortion);
    CONST_PTR(Distortion) getDistortion() const;
    
    
protected:
    /// Return a shared pointer to this
    Ptr getThisPtr() {
        return shared_from_this();
    }

    lsst::afw::geom::Box2I& getAllTrimmedPixels() {
        return _hasTrimmablePixels ? _trimmedAllPixels : _allPixels;
    }

   
    
private:
    Id _id;
    bool _isTrimmed;                    // Have all the bias/overclock regions been trimmed?
    lsst::afw::geom::Box2I _allPixels;          // Bounding box of all the Detector's pixels
    bool _hasTrimmablePixels;           // true iff Detector has pixels that can be trimmed (e.g. a CCD)
    double _pixelSize;                  // Size of a pixel in mm
    Orientation _orientation;           // orientation of this Detector
    FpPoint _center;           // position of _centerPixel (mm)
    lsst::afw::geom::Extent2D _size;            // Size in mm of this Detector
    lsst::afw::geom::Box2I _trimmedAllPixels;   // Bounding box of all the Detector's pixels after bias trimming
    boost::weak_ptr<Detector> _parent;  // Parent Detector in the hierarchy

    std::vector<lsst::afw::image::DefectBase::Ptr> _defects; // Defects in this detector
    
    CONST_PTR(Distortion) _distortion;
};

namespace detail {
    template<typename T>
    struct sortPtr :
        public std::binary_function<typename T::Ptr &, typename T::Ptr const&, bool> {
        bool operator()(typename T::Ptr &lhs, typename T::Ptr const& rhs) const {
            return *lhs < *rhs;
        }
    };
    /**
     * Rotate a BBox about the center of some larger region by a multiple of 90 degrees 
     */
    lsst::afw::geom::Box2I rotateBBoxBy90(
            lsst::afw::geom::Box2I const& bbox,         ///< the BBox to rotate
            int n90,                            ///< number of 90-degree anti-clockwise turns to make
            lsst::afw::geom::Extent2I const& dimensions ///< The size of the region wherein bbox dwells
    );
}


/**
 * @brief DetectorXYTransform: converts an XYTransform in FP coordinate system to detector coords
 *
 * Given a "global" XYTransform in the focal plane coordinate system, this class implements the 
 * coordinate transformation necessary to get the same XYTransform in the coordinate system of an 
 * individual detector.
 */
class DetectorXYTransform : public afw::geom::XYTransform
{
public:
    DetectorXYTransform(CONST_PTR(afw::geom::XYTransform) fpTransform, CONST_PTR(Detector) detector);
    virtual ~DetectorXYTransform() { }

    virtual PTR(afw::geom::XYTransform) clone() const;
    virtual PTR(afw::geom::XYTransform) invert() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeForwardTransform(Point2D const &pixel) const;
    virtual AffineTransform linearizeReverseTransform(Point2D const &pixel) const;

protected:
    CONST_PTR(XYTransform) _fpTransform;
    CONST_PTR(Detector) _detector;
};

    
}}}

#endif
