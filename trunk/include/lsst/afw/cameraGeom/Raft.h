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
 
#if !defined(LSST_AFW_CAMERAGEOM_RAFT_H)
#define LSST_AFW_CAMERAGEOM_RAFT_H

#include <string>
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/cameraGeom/DetectorMosaic.h"

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
class Raft : public DetectorMosaic {
public:
    typedef boost::shared_ptr<Raft> Ptr;
    typedef boost::shared_ptr<const Raft> ConstPtr;

    Raft(Id id,               ///< ID for Mosaic
         int const nCol,      ///< Number of columns of detectors
         int const nRow       ///< Number of rows of detectors
        )
        : DetectorMosaic(id, nCol, nRow) {}
    virtual ~Raft() {}

    double getPixelSize() const;
};

}}}

#endif
