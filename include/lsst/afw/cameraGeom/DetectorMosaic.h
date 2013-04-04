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
 
#if !defined(LSST_AFW_CAMERAGEOM_DETECTORMOSAIC_H)
#define LSST_AFW_CAMERAGEOM_DETECTORMOSAIC_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/cameraGeom/Orientation.h"
#include "lsst/afw/cameraGeom/FpPoint.h"

/**
 * @file
 *
 * Describe a Mosaic of Detectors
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Describe a set of Detectors that are physically closely related (e.g. on the same invar support)
 */
class DetectorMosaic : public Detector {
public:
    typedef boost::shared_ptr<DetectorMosaic> Ptr;
    typedef boost::shared_ptr<const DetectorMosaic> ConstPtr;

    typedef std::vector<Detector::Ptr> DetectorSet;
#if 0                                   // N.b. don't say "DetectorSet::iterator" for swig's sake
    typedef detectorSet::iterator iterator;
#else
    typedef std::vector<boost::shared_ptr<Detector> >::iterator iterator;
#endif
    typedef std::vector<Detector::Ptr>::const_iterator const_iterator;

    DetectorMosaic(Id id,               ///< ID for Mosaic
                   int const nCol,      ///< Number of columns of detectors
                   int const nRow       ///< Number of rows of detectors
                  ) : Detector(id, false), _nDetector(nCol, nRow) {}
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
    virtual void setCenter(FpPoint const& center);
    virtual FpExtent getSize() const;
    //
    // Add a Detector to the DetectorMosaic
    //
    void addDetector(lsst::afw::geom::Point2I const& index, lsst::afw::cameraGeom::FpPoint const& center,
                     Orientation const& orient, Detector::Ptr det);
    //
    // Find a Detector given an Id or pixel position
    //
    Detector::Ptr findDetector(Id const id) const;
    Detector::Ptr findDetectorMm(lsst::afw::cameraGeom::FpPoint const& posMm) const;
    //
    // Translate between physical positions in mm to pixels
    //
    virtual lsst::afw::geom::Point2D getPixelFromPosition(lsst::afw::cameraGeom::FpPoint const& pos) const;

private:
    DetectorSet _detectors;             // The Detectors that make up this DetectorMosaic
    std::pair<int, int> _nDetector;     // the number of columns/rows of Detectors
};

}}}

#endif
