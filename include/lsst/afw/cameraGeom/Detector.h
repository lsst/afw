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

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {
 
/**
 * Prefix for detector-specific coordinate systems
 *
 * A full detector-specific coordinate system combines this prefix with the detector name
 * as follows: <prefix>:<detector name>, e.g. "pixels:R11 S01".
 */
class DetectorSys : public BaseCoordSys {
public:
    explicit DetectorSys(std::string const &name) : BaseCoordSys(name) {}
    ~DetectorSys() {}
};


class Detector {
    /**
     * Make a Detector
     *
     * Warning: the keys for the detector-specific coordinate systems in the transform registry
     * must include the detector name.
     */
    explicit Detector(
        std::string const &name,    ///< detector name
        transformRegistry const &geom::TransformRegistry  ///< transform registry for this detector
    ) _name(name), _transformRegistry(transformRegistry) {}

    ~Detector() {}

    /** make a CoordSys from another CoordSys; a no-op */
    CoordSys makeCoordSys(CoordSys const &coordSys) const { return coordSys; }

    /** make a CoordSys from a DetectorSys prefix */
    CoordSys makeCoordSys(DetectorSys const &detectorSys) const {
        std::ostringstream os;
        os << DetectorSys.getName() << ":" << getName();
        return CoordSys(os.str())
    }

    /**
     * Convert a CoordPoint2 from one coordinate system to another specified by a CoordSys
     */
    CoordPoint2 convert(
        CoordPoint2 const &fromPoint,   ///< point to convert
        CoordSys const &toSys           ///< coordinate system to which to convert,
                                        ///< e.g. FOCAL_PLANE or detector.getCoordSys(PIXELS)
    ) const {
        return _transformRegistry.convert(fromPoint, toSys);
    }

    /**
     * Convert a CoordPoint2 from one coordinate system to another specified by a DetectorSys
     */
    CoordPoint2 convert(
        CoordPoint2 const &fromPoint,   ///< point to convert
        DetectorSys const &toSys        ///< detector-specific system to which to convert, e.g. PIXELS
    ) const {
        CoordSys toCoordSys = getCoordSys(toSys);
        return convert(fromPoint, toCoordSys);
    }

    /** Get the detector name */
    std::string getName() { return _name; }

    /** Get the transform registry */
    geom::TransformRegistry getTransformRegistry() { return _transformRegistry; }

private:
    std::string _name; ///< detector name
    geom::TransformRegistry _transformRegistry; ///< registry of coordinate transforms
}

/**
 * Nominal pixels on the detector (unbinned)
 * This ignores manufacturing imperfections, "tree ring" distortions and all other such effects.
 * It is a uniform grid of rectangular (usually square) pixels.
 */
DetectorSys const PIXELS("pixels");

/**
 * The actual pixels where the photon lands and electrons are generated (unbinned)
 * This takes into account manufacturing defectos, "tree ring" distortions and other such effects.
 */
DetectorSys const ACTUAL_PIXELS("pixels");

}}}

#endif
