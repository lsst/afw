#if !defined(LSST_AFW_CAMERAGEOM_DETECTORMOSAIC_H)
#define LSST_AFW_CAMERAGEOM_DETECTORMOSAIC_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"

/**
 * @file
 *
 * Describe a Mosaic of Detectors
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

namespace afwGeom = lsst::afw::geom;

/**
 * Describe a detector's orientation
 */
class Orientation {
public:
    explicit Orientation(double pitch=0.0, ///< pitch (rotation in YZ), radians
                         double roll=0.0,  ///< roll (rotation in XZ), radians
                         double yaw=0.0) : ///< yaw (rotation in XY), radians
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
 * Describe the layout of Detectors in a DetectorMosaic
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
class DetectorMosaic : public Detector {
public:
    typedef boost::shared_ptr<DetectorMosaic> Ptr;
    typedef boost::shared_ptr<const DetectorMosaic> ConstPtr;

    typedef std::vector<DetectorLayout::Ptr> DetectorSet;
#if 0                                   // N.b. don't say "DetectorSet::iterator" for swig's sake
    typedef detectorSet::iterator iterator;
#else
    typedef std::vector<boost::shared_ptr<DetectorLayout> >::iterator iterator;
#endif
    typedef std::vector<DetectorLayout::Ptr>::const_iterator const_iterator;

    DetectorMosaic(Id id) : Detector(id, false), _nDetector(0, 0) {
        ;
    }
    virtual ~DetectorMosaic() {}
    //
    // Provide iterators for all the Ccd's Detectors
    //
    iterator begin() { return _detectors.begin(); }
    const_iterator begin() const { return _detectors.begin(); }
    iterator end() { return _detectors.end(); }
    const_iterator end() const { return _detectors.end(); }
    //
    // Geometry of Detector --- i.e. mm not pixels
    //
    virtual lsst::afw::geom::Extent2D getSize() const;
    //
    // Add a Detector to the DetectorMosaic
    //
    void addDetector(afwGeom::Point2I const& index, afwGeom::Point2D const& center,
                     cameraGeom::Orientation const& orient, cameraGeom::Detector::Ptr det);
    //
    // Find a Detector given an Id or pixel position
    //
    DetectorLayout::Ptr findDetector(Id const id) const;
    DetectorLayout::Ptr findDetector(lsst::afw::geom::Point2I const& pixel) const;
private:
    DetectorSet _detectors;             // The Detectors that make up this DetectorMosaic
    std::pair<int, int> _nDetector;     // the number of columns/rows of Detectors
};

}}}

#endif
