#if !defined(LSST_AFW_CAMERAGEOM_RAFT_H)
#define LSST_AFW_CAMERAGEOM_RAFT_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"

/**
 * @file
 *
 * Describe a Raft of Detectors
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Describe a detector's orientation
 */
class Orientation {
public:
    explicit Orientation(double pitch=0.0, ///< pitch, UNITS
                         double roll=0.0,  ///< roll, UNITS
                         double yaw=0.0) : ///< yaw, UNITS
        _pitch(pitch), _roll(roll), _yaw(yaw) {}
    /// Return the pitch angle
    double getPitch() const { return _pitch; }
    /// Return the roll angle
    double getRoll() const { return _roll; }
    /// Return the yaw angle
    double getYaw() const { return _yaw; }
private:
    double _pitch;                         // pitch
    double _roll;                          // roll
    double _yaw;                           // yaw
};

/************************************************************************************************************/
/**
 * Describe the layout of Detectors in a Raft
 */    
class DetectorLayout {
public:
    typedef boost::shared_ptr<DetectorLayout> Ptr;
    typedef boost::shared_ptr<const DetectorLayout> ConstPtr;

    explicit DetectorLayout(Detector::Ptr detector,         ///< The detector
                            Orientation const& orientation, ///< the detector's orientation
                            lsst::afw::geom::Point2D center, ///< the detector's center
                            lsst::afw::geom::Point2I origin  ///< The Detector's approximate pixel origin
                           )
        : _detector(detector), _orientation(orientation), _center(center), _origin(origin) {}

    /// Return the Detector
    Detector::Ptr getDetector() const { return _detector; }
    /// Return the Detector's Orientation
    Orientation getOrientation() const { return _orientation;}
    /// Return the Detector's center
    lsst::afw::geom::Point2D getCenter() const { return _center; }
    /// Return the Detector's origin
    lsst::afw::geom::Point2I getOrigin() const { return _origin; }
private:
    Detector::Ptr _detector;
    Orientation _orientation;
    lsst::afw::geom::Point2D _center;
    lsst::afw::geom::Point2I _origin;
};

/**
 * Describe a set of Detectors that are physically closely related (e.g. on the same invar support)
 */
class Raft {
public:
    typedef boost::shared_ptr<Raft> Ptr;
    typedef boost::shared_ptr<const Raft> ConstPtr;

    typedef std::vector<DetectorLayout::Ptr> DetectorSet;
#if 0                                   // N.b. don't say "DetectorSet::iterator" for swig's sake
    typedef detectorSet::iterator iterator;
#else
    typedef std::vector<boost::shared_ptr<DetectorLayout> >::iterator iterator;
#endif
    typedef std::vector<DetectorLayout::Ptr>::const_iterator const_iterator;

    Raft(Id id) : _id(id), _allPixels(), _nDetector(0, 0) {
        ;
    }
    virtual ~Raft() {}
    //
    // Provide iterators for all the Ccd's Detectors
    //
    iterator begin() { return _detectors.begin(); }
    const_iterator begin() const { return _detectors.begin(); }
    iterator end() { return _detectors.end(); }
    const_iterator end() const { return _detectors.end(); }

    /// Return the Detector's Id
    Id getId() const { return _id; }

    /// Return Raft's total footprint
    lsst::afw::image::BBox& getAllPixels() {
        return _allPixels;
    }
    lsst::afw::image::BBox const& getAllPixels() const {
        return _allPixels;
    }
    //
    // Geometry of Detector --- i.e. mm not pixels
    //

    /// Return size in mm of this Raft
    lsst::afw::geom::Extent2D getSize() const;
    //
    // Add a Detector to the Raft
    //
    void addDetector(int const iX, int const iY, Detector::Ptr det);
    //
    // Find a Detector given an Id or pixel position
    //
    DetectorLayout::Ptr findDetector(Id const id) const;
    DetectorLayout::Ptr findDetector(lsst::afw::geom::Point2I const& pixel) const;
private:
    Id _id;
    lsst::afw::image::BBox _allPixels;  // Bounding box of all the Raft's pixels
    DetectorSet _detectors;             // The Detectors that make up this Raft
    std::pair<int, int> _nDetector;     // the number of columns/rows of Detectors
};

}}}

#endif
