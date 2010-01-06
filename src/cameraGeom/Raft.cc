/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/Raft.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace camGeom = lsst::afw::cameraGeom;

/************************************************************************************************************/
/**
 * Add an Detector to the set known to be part of this Raft
 *
 *  The \c iX and \c iY values are the 0-indexed position of the Detector in the Raft; e.g. (0, 2)
 * for the top left Detector in a 3x3 raft
 */
void camGeom::Raft::addDetector(
        int const iX,                   ///< x-index of this Detector
        int const iY,                   ///< y-index of this Detector
        camGeom::Detector::Ptr det      ///< The detector to add to the Raft's manifest.
                               )
{
    bool const isTrimmed = true;        // We always work in trimmed coordinates at the Raft level
    //
    // Correct Detector's coordinate system to be absolute within Raft
    //
    afwImage::BBox detPixels = det->getAllPixels(isTrimmed);
    detPixels.shift(iX*detPixels.getWidth(), iY*detPixels.getHeight());

    getAllPixels().grow(detPixels.getLLC());
    getAllPixels().grow(detPixels.getURC());
    
    camGeom::Orientation orient(0.0, 0.0, 0.0);
    afwGeom::Point2D center;
    afwGeom::Point2I origin = afwGeom::Point2I::makeXY(iX*detPixels.getWidth(), iY*detPixels.getHeight());
    camGeom::DetectorLayout::Ptr detL(new camGeom::DetectorLayout(det, orient, center, origin));

    _detectors.push_back(detL);
}

/************************************************************************************************************/

namespace {
    struct findById {
        findById(camGeom::Id id) : _id(id) {}
        bool operator()(camGeom::DetectorLayout::Ptr det) const {
            return _id == det->getDetector()->getId();
        }
    private:
        camGeom::Id _id;
    };

    struct findByPos {
        findByPos(
                  afwGeom::Point2I point
                 ) :
            _point(afwImage::PointI(point[0], point[1]))
        { }

        bool operator()(camGeom::DetectorLayout::Ptr det) const {
            afwImage::PointI relPoint = _point;
            relPoint.shift(-det->getOrigin()[0], -det->getOrigin()[1]);

            return det->getDetector()->getAllPixels().contains(relPoint);
        }
    private:
        afwImage::PointI _point;
    };
}

/**
 * Find an Detector given an Id
 */
camGeom::DetectorLayout::Ptr camGeom::Raft::findDetector(camGeom::Id const id) const {
    DetectorSet::const_iterator result = std::find_if(_detectors.begin(), _detectors.end(), findById(id));
    if (result == _detectors.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("Unable to find Detector with serial %||") % id).str());
    }
    return *result;
}

/**
 * Find an Detector given a position
 */
camGeom::DetectorLayout::Ptr camGeom::Raft::findDetector(afwGeom::Point2I const& pixel // the desired pixel
                                                        ) const {
    DetectorSet::const_iterator result = std::find_if(_detectors.begin(), _detectors.end(), findByPos(pixel));
    if (result == _detectors.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("Unable to find Detector containing pixel (%d, %d)") %
                           pixel.getX() % pixel.getY()).str());
    }
    return *result;
}
