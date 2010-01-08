/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/DetectorMosaic.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace cameraGeom = lsst::afw::cameraGeom;

/************************************************************************************************************/
/**
 * Return a DetectorMosaic's size in mm
 */
afwGeom::Extent2D cameraGeom::DetectorMosaic::getSize() const {
    afwGeom::Extent2D size(0.0);        // the desired size
    // This code can probably use afwGeom's bounding boxes when the are available
    afwGeom::Point2D LLC, URC;          // corners of DetectorMosaic

    for (cameraGeom::DetectorMosaic::const_iterator detL = begin(), end = this->end(); detL != end; ++detL) {
        cameraGeom::Detector::Ptr det = (*detL)->getDetector();
        afwGeom::Extent2D detectorSize = det->getSize();

        double const yaw = (*detL)->getOrientation().getYaw();
        if (yaw != 0.0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RangeErrorException,
                              (boost::format("(yaw == %f) != 0 is not supported for Detector %||") %
                               yaw % det->getId()).str());
        }

        if (detL == begin()) {           // first detector
            LLC = (*detL)->getCenter() - detectorSize/2;
            URC = (*detL)->getCenter() + detectorSize/2;
        } else {
            afwGeom::Point2D llc = (*detL)->getCenter() - detectorSize/2; // corners of this Detector
            afwGeom::Point2D urc = (*detL)->getCenter() + detectorSize/2;

            if (llc[0] < LLC[0]) {      // X
                LLC[0] = llc[0];
            }
            if (llc[1] < LLC[1]) {      // Y
                LLC[1] = llc[1];
            }

            if (urc[0] > URC[0]) {      // X
                URC[0] = urc[0];
            }
            if (urc[1] > URC[1]) {      // Y
                URC[1] = urc[1];
            }
        }
    }

    return URC - LLC;
}

/**
 * Add an Detector to the set known to be part of this DetectorMosaic
 *
 *  The \c index is the 0-indexed position of the Detector in the DetectorMosaic; e.g. (0, 2)
 * for the top left Detector in a 3x3 detectormosaic
 */
void cameraGeom::DetectorMosaic::addDetector(
        afwGeom::Point2I const& index, ///< index of this Detector in DetectorMosaic thought of as a grid
        afwGeom::Point2D const& center, ///< center of this Detector wrt center of DetectorMosaic
        cameraGeom::Orientation const& orient, ///< orientation of this Detector
        cameraGeom::Detector::Ptr det   ///< The detector to add to the DetectorMosaic's manifest.
                                            )
{
    bool const isTrimmed = true;        // We always work in trimmed coordinates at the DetectorMosaic level
    int const iX = index[0];
    int const iY = index[1];
    //
    // Correct Detector's coordinate system to be absolute within DetectorMosaic
    //
    afwImage::BBox detPixels = det->getAllPixels(isTrimmed);
    detPixels.shift(iX*detPixels.getWidth(), iY*detPixels.getHeight());

    getAllPixels().grow(detPixels.getLLC());
    getAllPixels().grow(detPixels.getURC());
    
    afwGeom::Point2I origin = afwGeom::Point2I::makeXY(iX*detPixels.getWidth(), iY*detPixels.getHeight());
    cameraGeom::DetectorLayout::Ptr detL(new cameraGeom::DetectorLayout(det, orient, center, origin));

    _detectors.push_back(detL);

    if (iX >= _nDetector.first) {
        _nDetector.first = iX + 1;
    }
    if (iY >= _nDetector.second) {
        _nDetector.second = iY + 1;
    }
}

/************************************************************************************************************/

namespace {
    struct findById {
        findById(cameraGeom::Id id) : _id(id) {}
        bool operator()(cameraGeom::DetectorLayout::Ptr det) const {
            return _id == det->getDetector()->getId();
        }
    private:
        cameraGeom::Id _id;
    };

    struct findByPos {
        findByPos(
                  afwGeom::Point2I point
                 ) :
            _point(afwImage::PointI(point[0], point[1]))
        { }

        bool operator()(cameraGeom::DetectorLayout::Ptr det) const {
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
cameraGeom::DetectorLayout::Ptr cameraGeom::DetectorMosaic::findDetector(
        cameraGeom::Id const id         ///< The desired ID
                                                                        ) const {
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
cameraGeom::DetectorLayout::Ptr cameraGeom::DetectorMosaic::findDetector(
        afwGeom::Point2I const& pixel   ///< the desired pixel
                                                                        ) const {
    DetectorSet::const_iterator result = std::find_if(_detectors.begin(), _detectors.end(), findByPos(pixel));
    if (result == _detectors.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("Unable to find Detector containing pixel (%d, %d)") %
                           pixel.getX() % pixel.getY()).str());
    }
    return *result;
}
