/* 
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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
 
#if !defined(LSST_AFW_CAMERAGEOM_CAMERASYS_H)
#define LSST_AFW_CAMERAGEOM_CAMERASYS_H

#include <string>
#include <sstream>
#include "lsst/afw/geom/TransformRegistry.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {
 
/**
 * Standard coordinate systems for CameraGeom
 * (see Detector.h for standard detector-specific coordinate system prefixes)
 */

/**
 * Focal plane coordinates:
 * Rectilinear x, y (and z when talking about the location of a detector) on the camera focal plane (mm).
 * For z=0 choose a convenient point near the focus at x, y = 0.
 */
CoordSys const geom::CoordSys("focalPlane") FOCAL_PLANE;

/**
 * Pupil coordinates:
 * Angular x,y offset from the vertex at the pupil (arcsec).
 */
CoordSys const geom::CoordSys("pupil") PUPIL;
    
}}}

#endif
