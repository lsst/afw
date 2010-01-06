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
 * Describe a set of Detectors that are physically closely related (e.g. on the same invar support)
 */
class Raft {
public:
    typedef std::vector<Detector> DetectorSet;
    typedef std::vector<Detector>::iterator iterator; // don't say "DetectorSet::iterator" for swig's sake
    typedef std::vector<Detector>::const_iterator const_iterator;

    Raft(Id id) : _id(id), _allPixels() {
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

    /// Return size in mm of this Raft
    lsst::afw::geom::Extent2D getSize() const {
        Eigen::Vector2d size;
        size << 0.0, 0.0;
        return lsst::afw::geom::Extent2D(size);
    }
    //
    // Geometry of Detector --- i.e. mm not pixels
    //

    //
    // Add a Detector to the Raft
    //
    void addDetector(int const iX, int const iY, Detector const& det);
    //
    // Find a Detector given an Id or pixel position
    //
    Detector getDetector(Id const id) const;
    Detector getDetector(lsst::afw::geom::Point2I const& pixel) const;
private:
    Id _id;
    lsst::afw::image::BBox _allPixels;  // Bounding box of all the Raft's pixels
    DetectorSet _detectors;             // The Detectors that make up this Raft
};

}}}

#endif
