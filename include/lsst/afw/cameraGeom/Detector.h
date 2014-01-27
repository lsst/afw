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
#include "lsst/afw/cameraGeom/CameraSys.h"
#include "lsst/afw/cameraGeom/CameraPoint.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

class Detector {
public:
    /**
     * Make a Detector
     *
     * @todo: Replace this constructor with one that takes an orientation
     * and construct the FocalPlane->Pixels entry from that Orientation
     *
     * @warning The keys for the detector-specific coordinate systems in the transform registry
     * must include the detector name.
     */
    explicit Detector(
        std::string const &name,    ///< detector name
        std::string const &serial,  ///< detector serial "number" that identifies the physical detector
        CameraTransformRegistry const &transformRegistry ///< transform registry for this detector
    ) : _name(name), _serial(serial), _transformRegistry(transformRegistry) {}

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
     * Convert a CameraPoint from one coordinate system to another
     *
     * @throw: pexExcept::InvalidParameterException if from or to coordinate system is unknown
     */
    CameraPoint convert(
        CameraPoint const &fromPoint,   ///< camera point to convert
        CameraSys const &toSys          ///< coordinate system to which to convert;
                                        ///< may be a full system or a detector prefix such as PIXELS
    ) const {
        CameraSys fullToSys = getCameraSys(toSys);
        geom::Point2D toPoint = _transformRegistry.convert(fromPoint.getPoint(), fromPoint.getCameraSys(), fullToSys);
        return CameraPoint(toPoint, fullToSys);
    }

    /** Get the detector name */
    std::string getName() const { return _name; }

    /** Get the detector serial "number" */
    std::string getSerial() const { return _serial; }

    /** Get the transform registry */
    CameraTransformRegistry getTransformRegistry() const { return _transformRegistry; }

    /**
     * Make a CameraPoint from a point and a camera system or detector prefix
     */
    CameraPoint makeCameraPoint(
        geom::Point2D point,    ///< 2-d point
        CameraSys cameraSys     ///< coordinate system; may be a full system or a detector prefix
    ) const {
        return CameraPoint(point, getCameraSys(cameraSys));
    }

private:
    std::string _name; ///< detector name
    std::string _serial;    ///< detector serial "number" that identifies the physical detector
    CameraTransformRegistry _transformRegistry; ///< registry of coordinate transforms
};

}}}

#endif
