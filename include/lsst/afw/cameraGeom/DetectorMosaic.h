#if !defined(LSST_AFW_CAMERAGEOM_DETECTORMOSAIC_H)
#define LSST_AFW_CAMERAGEOM_DETECTORMOSAIC_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/cameraGeom/Orientation.h"

/**
 * @file
 *
 * Describe a Mosaic of Detectors
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

namespace afwGeom = lsst::afw::geom;

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
                            afwGeom::Point2D center, ///< the detector's center
                            afwGeom::Point2I origin  ///< The Detector's approximate pixel origin
                           )
        : _detector(detector), _origin(origin) {
        detector->setOrientation(orientation);
        detector->setCenter(center);
    }

    /// Return the Detector
    Detector::Ptr getDetector() const { return _detector; }
    /// Return the Detector's origin
    afwGeom::Point2I getOrigin() const { return _origin; }
private:
    Detector::Ptr _detector;
    afwGeom::Point2I _origin;
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
    virtual void setCenter(afwGeom::Point2D const& center);
    virtual afwGeom::Extent2D getSize() const;
    //
    // Add a Detector to the DetectorMosaic
    //
    void addDetector(afwGeom::Point2I const& index, afwGeom::Point2D const& center,
                     Orientation const& orient, Detector::Ptr det);
    //
    // Find a Detector given an Id or pixel position
    //
    DetectorLayout::Ptr findDetector(Id const id) const;
    DetectorLayout::Ptr findDetector(afwGeom::Point2I const& pixel) const;
    DetectorLayout::Ptr findDetector(afwGeom::Point2D const& posMm) const;
    //
    // Translate between physical positions in mm to pixels
    //
    virtual afwGeom::Point2I getIndexFromPosition(afwGeom::Point2D pos) const;
    virtual afwGeom::Point2D getPositionFromIndex(afwGeom::Point2I pix) const;
    virtual afwGeom::Point2D getPositionFromIndex(afwGeom::Point2I pix, bool const) const {
        return getPositionFromIndex(pix);
    }
private:
    DetectorSet _detectors;             // The Detectors that make up this DetectorMosaic
    std::pair<int, int> _nDetector;     // the number of columns/rows of Detectors
};

}}}

#endif
