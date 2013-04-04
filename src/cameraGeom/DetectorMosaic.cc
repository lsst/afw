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
 
/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/DetectorMosaic.h"
#include "lsst/afw/cameraGeom/FpPoint.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace cameraGeom = lsst::afw::cameraGeom;

/// Set the DetectorMosaic's center
void cameraGeom::DetectorMosaic::setCenter(cameraGeom::FpPoint const& center) {
    cameraGeom::Detector::setCenter(center);
    //
    // Update the centers for all children too
    //
    for (cameraGeom::DetectorMosaic::const_iterator ptr = begin(), end = this->end(); ptr != end; ++ptr) {
        (*ptr)->setCenter((*ptr)->getCenter() + center);
    }
}

/************************************************************************************************************/
/**
 * Return a DetectorMosaic's size in mm
 */
cameraGeom::FpExtent cameraGeom::DetectorMosaic::getSize() const {
    afwGeom::Extent2D size(0.0);        // the desired size
    // This code can probably use afwGeom's bounding boxes when they are available
    afwGeom::Point2D LLC, URC;          // corners of DetectorMosaic

    for (cameraGeom::DetectorMosaic::const_iterator ptr = begin(), end = this->end(); ptr != end; ++ptr) {
        cameraGeom::Detector::Ptr det = *ptr;
        afwGeom::Extent2D detectorHalfSize(det->getSize().getMm());
        detectorHalfSize /= 2;
        
        double const yaw = det->getOrientation().getYaw();
        if (yaw != 0.0) {
            throw LSST_EXCEPT(pexExcept::RangeErrorException,
                              (boost::format("(yaw == %f) != 0 is not supported for Detector %||") %
                               yaw % det->getId()).str());
        }

        if (ptr == begin()) {           // first detector
            LLC = det->getCenter().getMm() - detectorHalfSize;
            URC = det->getCenter().getMm() + detectorHalfSize;
        } else {
            afwGeom::Point2D llc = det->getCenter().getMm() - detectorHalfSize; // corners of this Detector
            afwGeom::Point2D urc = det->getCenter().getMm() + detectorHalfSize;

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

    return cameraGeom::FpExtent(URC - LLC);
}

/**
 * Add an Detector to the set known to be part of this DetectorMosaic
 *
 *  The \c index is the 0-indexed position of the Detector in the DetectorMosaic; e.g. (0, 2)
 * for the top left Detector in a 3x3 mosaic
 */
void cameraGeom::DetectorMosaic::addDetector(
        afwGeom::Point2I const& index, ///< index of this Detector in DetectorMosaic thought of as a grid
        cameraGeom::FpPoint const& center, ///< center of this Detector wrt center of DetectorMosaic
        cameraGeom::Orientation const& orient, ///< orientation of this Detector
        cameraGeom::Detector::Ptr det   ///< The detector to add to the DetectorMosaic's manifest.
                                            )
{
    bool const isTrimmed = true;        // We always work in trimmed coordinates at the DetectorMosaic level
    int const iX = index[0];
    int const iY = index[1];
    
    if (iX < 0 || iX >= _nDetector.first) {
        throw LSST_EXCEPT(pexExcept::RangeErrorException,
                          (boost::format("Col index %d is not in range 0..%d for Detector %||") %
                           iX % _nDetector.first % det->getId()).str());
    }
    if (iY < 0 || iY >= _nDetector.second) {
        throw LSST_EXCEPT(pexExcept::RangeErrorException,
                          (boost::format("Row index %d is not in range 0..%d for Detector %||") %
                           iY % _nDetector.second % det->getId()).str());
    }
    //
    // Don't permit non-square Detectors to have relative rotations other than 0, 180
    //
    /*
    if (_detectors.size() > 0) {
        if ((orient.getNQuarter() - (*begin())->getOrientation().getNQuarter())%2 != 0 &&
            det->getAllPixels(true).getWidth() != det->getAllPixels(true).getHeight()) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                              (boost::format("Rotation of detector %|| (nQuarter == %d) "
                                             "is incompatible with %|| (nQuarter == %d)") %
                               det->getId() % orient.getNQuarter() %
                               (*begin())->getId() % (*begin())->getOrientation().getNQuarter()).str());
        }
    }
    */
    //
    // Correct Detector's coordinate system to be absolute within DetectorMosaic
    //
    afwGeom::Box2I detPixels = det->getAllPixels(isTrimmed);
    detPixels.shift(
        geom::Extent2I(iX*detPixels.getWidth(), iY*detPixels.getHeight())
    );
    getAllPixels().include(detPixels);
    
    // In order to use the 'grid' location method (i.e. iX*width), rotation must be after getWidth/Height calls.
    det->setOrientation(orient);
    
    afwGeom::Point2D centerPixel(
        iX*detPixels.getWidth() + detPixels.getWidth()/2,
        iY*detPixels.getHeight() + detPixels.getHeight()/2
    );
    det->setCenter(center);

    // insert new Detector, keeping the Detectors sorted
    _detectors.insert(
        std::lower_bound(
            _detectors.begin(), 
            _detectors.end(), 
            det, 
            cameraGeom::detail::sortPtr<Detector>()
        ), 
        det
    );
    det->setParent(getThisPtr());
}

/************************************************************************************************************/

namespace {
    struct findById {
        findById(cameraGeom::Id id) : _id(id) {}
        bool operator()(cameraGeom::Detector::Ptr det) const {
            return _id == det->getId();
        }
    private:
        cameraGeom::Id _id;
    };

    struct findByMm {
        findByMm(
                 cameraGeom::FpPoint point
                 )
            :
            _point(point)
        { }

        /*
         * Does point lie with square footprint of detector, once we allow for its rotation?
         */
        bool operator()(cameraGeom::Detector::Ptr det) const {
            afwGeom::Extent2D off = afwGeom::Extent2D((_point - det->getCenter()).getMm());
            double const c = det->getOrientation().getCosYaw();
            double const s = det->getOrientation().getSinYaw();

            double const dx = off[0]*c - off[1]*s; // rotate into CCD frame
            double const xSize2 = det->getSize().getMm()[0]/2;  // xsize/2
            if (dx < -xSize2 || dx > xSize2) {
                return false;
            }

            double const dy = off[0]*s + off[1]*c; // rotate into CCD frame
            double const ySize2 = det->getSize().getMm()[1]/2;  // ysize/2
            if (dy < -ySize2 || dy > ySize2) {
                return false;
            }

            return true;
        }
    private:
        cameraGeom::FpPoint _point;
    };
}

/**
 * Find an Detector given an Id
 */
cameraGeom::Detector::Ptr cameraGeom::DetectorMosaic::findDetector(
        cameraGeom::Id const id         ///< The desired ID
) const {
    DetectorSet::const_iterator result = 
        std::find_if(_detectors.begin(), _detectors.end(), findById(id));
    if (result == _detectors.end()) {
        throw LSST_EXCEPT(pexExcept::OutOfRangeException,
                          (boost::format("Unable to find Detector with serial %||") % id).str());
    }
    return *result;
}

/**
 * Find an Detector given a physical position in mm
 */
cameraGeom::Detector::Ptr cameraGeom::DetectorMosaic::findDetectorMm(
        cameraGeom::FpPoint const& pos     ///< the desired position; mm from the centre
) const {
    DetectorSet::const_iterator result = 
        std::find_if(_detectors.begin(), _detectors.end(), findByMm(pos));
    if (result == _detectors.end()) {
        throw LSST_EXCEPT(pexExcept::OutOfRangeException,
                          (boost::format("Unable to find Detector containing pixel (%g, %g)") %
                           pos.getMm().getX() % pos.getMm().getY()).str());
    }
    return *result;
}

/**
 * Return the pixel position given an offset from the mosaic centre, in mm

 */
afwGeom::Point2D cameraGeom::DetectorMosaic::getPixelFromPosition(
        cameraGeom::FpPoint const& pos     ///< Offset from mosaic centre, mm
                                                                 ) const
{
    cameraGeom::Detector::ConstPtr det = findDetectorMm(pos);
    return det->getPixelFromPosition(pos);
}

