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
#include "lsst/base.h"
#include "lsst/afw/geom/TransformRegistry.h"
#include "lsst/afw/cameraGeom/Amplifier.h"
#include "lsst/afw/cameraGeom/CameraSys.h"
#include "lsst/afw/cameraGeom/CameraPoint.h"
#include "lsst/afw/cameraGeom/Orientation.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Type of imaging detector
 */
enum DetectorType {
    SCIENCE,
    FOCUS,
    GUIDER,
    WAVEFRONT,
};


/**
 * Information about a CCD or other imaging detector
 *
 * Supports conversion of CameraPoint between FOCAL_PLANE and pixel-based coordinate systems.
 * Also an iterator over amplifiers (in C++ use begin(), end(), in Python use "for amplifier in detector").
 *
 * @warning Only supports detectors with square pixels
 */
class Detector {
public:
    typedef std::vector<CONST_PTR(Amplifier)> AmplifierList;
    /**
     * Make a Detector
     *
     * @warning
     * * The keys for the detector-specific coordinate systems in the transform registry
     *   must include the detector name (even though this is redundant).
     */
    explicit Detector(
        std::string const &name,    ///< name of detector's location in the camera
        DetectorType type,          ///< type of detector
        std::string const &serial,  ///< serial "number" that identifies the physical detector
        AmplifierList const &amplifierList, ///< list of amplifier data
        Orientation const &orientation,     ///< detector position and orientation in focal plane
        double pixelSize,           ///< size of pixel along x or y (mm); pixels are assumed to be square
        CameraTransformMap const &transformMap  ///< list of coordinate transforms for this detector
    );

    ~Detector() {}

    /**
     * Convert a CameraPoint from one coordinate system to another
     *
     * @throw pexExcept::InvalidParameterException if from or to coordinate system is unknown
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

    /** 
     * Get a coordinate system from a coordinate system (return input unchanged)
     */
    CameraSys const getCameraSys(CameraSys const &cameraSys) const { return cameraSys; }

    /** 
     * Get a coordinate system from a detector system prefix (add detector name)
     */
    CameraSys const getCameraSys(CameraSysPrefix const &cameraSysPrefix) const {
        return CameraSys(cameraSysPrefix.getSysName(), _name);
    }

    /** Get the detector name */
    std::string getName() const { return _name; }

    DetectorType getType() const { return _type; }

    /** Get the detector serial "number" */
    std::string getSerial() const { return _serial; }

    /** Get the transform registry */
    CameraTransformRegistry const getTransformRegistry() const { return _transformRegistry; }

    /** Get iterator to beginning of amplifier list */
    AmplifierList::const_iterator begin() const { return _amplifierList.begin(); }

    /** Get iterator to end of amplifier list */
    AmplifierList::const_iterator end() const { return _amplifierList.end(); }

    /**
     * Get the amplifier specified by index
     *
     * @throw something if index is out of range (uses "at" internally)
     */
    CONST_PTR(Amplifier) operator[](size_t i) const { return _amplifierList.at(i); }

    /**
     * Get the amplifier specified by name
     *
     * @throw lst::pex::exceptions::InvalidParameterException if no such amplifier
     */
    CONST_PTR(Amplifier) operator[](std::string const &name) const;

    /**
     * Get number of amplifiers. Renamed to __len__ in Python.
     */
    size_t size() const {return _amplifierList.size(); }

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
    typedef boost::unordered_map<std::string, CONST_PTR(Amplifier)> _AmpMap;
    // set _amplifierMap from _amplifierList
    void _makeAmplifierMap();

    std::string _name;      ///< name of detector's location in the camera
    DetectorType _type;     ///< type of detector
    std::string _serial;    ///< serial "number" that identifies the physical detector
    AmplifierList _amplifierList; ///< list of amplifier data
    _AmpMap _amplifierMap;  ///< map of amplifier name: amplifier
    Orientation _orientation;       ///< position and orientation of detector in focal plane
    double _pixelSize;      ///< size of pixel along x or y (mm)
    CameraTransformRegistry _transformRegistry; ///< registry of coordinate transforms
};

}}}

#endif
