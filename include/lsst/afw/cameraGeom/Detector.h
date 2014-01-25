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
 
#if !defined(LSST_AFW_CAMERAGEOM_DETECTOR_H)
#define LSST_AFW_CAMERAGEOM_DETECTOR_H

#include <string>
#include <sstream>
#include "lsst/afw/geom/TransformRegistry.h"
#include "lsst/afw/geom/CoordPoint2.h"
#include "lsst/afw/cameraGeom/CameraSys.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

class Detector {
    /**
     * Make a Detector
     *
     * Warning: the keys for the detector-specific coordinate systems in the transform registry
     * must include the detector name.
     */
    explicit Detector(
        std::string const &name,    ///< detector name
        &geom::TransformRegistry<CameraSys> const transformRegistry ///< transform registry for this detector
    ) _name(name), _transformRegistry(transformRegistry) {}

    ~Detector() {}

    /** 
     * Get a coordinate system from a coordinate system (return input unchanged)
     */
    CameraSys getCameraSys(CameraSys const &cameraSys) const { return cameraSys; }

    /** 
     * Get a coordinate system from a detector system prefix (add detector name)
     */
    CameraSys getCameraSys(DetectorSysPrefix const &detectorSysPrefix) const {
        return CameraSys(detectorSysPrefix.getSysName(), _name);
    }

    /**
     * Convert a CoordPoint2 from one coordinate system to adetectorSysPrefixnother specified by a CameraSys
     */
    CoordPoint2 convert(
        CoordPoint2 const &fromPoint,   ///< point to convert
        CameraSys const &toSys        ///< detector-specific system to which to convert, e.g. PIXELS
    ) const {
        return convert(fromPoint, getCameraSys(toSys));
    }

    /** Get the detector name */
    std::string getName() { return _name; }

    /** Get the transform registry */
    geom::TransformRegistry<CameraSys> getTransformRegistry() { return _transformRegistry; }

private:
    std::string _name; ///< detector name
    geom::TransformRegistry<CameraSys> _transformRegistry; ///< registry of coordinate transforms
}

}}}

#endif
