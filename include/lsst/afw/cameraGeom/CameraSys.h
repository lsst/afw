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

#include <functional>
#include <string>
#include <ostream>
#include <sstream>
#include "lsst/afw/geom/TransformRegistry.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Base class for camera coordinate systems
 *
 * This version has no detector name and used by Detector.makeCameraSys
 * to construct a fully specified CameraSys
 *
 * This is Jim Bosch's clever idea for simplifying Detector.convert;
 * CameraSys is always complete and BaseCameraSys is not.
 */
class BaseCameraSys {
public:
    explicit BaseCameraSys(
        std::string const &sysName  ///< coordinate system name
    ) : _sysName(sysName) {}
    ~BaseCameraSys() {}

    /**
     * Get coordinate system name
     */
    std::string getSysName() const { return _sysName; };

    bool operator==(BaseCameraSys const &rhs) const {
        return _sysName == rhs.getSysName();
    }

    bool operator!=(BaseCameraSys const &rhs) const {
        return !(*this == rhs);
    }
private:
    std::string _sysName;   ///< coordinate system name
};

/**
 * Base class for coordinate system keys used in in TransformRegistry
 *
 * @note A subclass is used for keys in TransformRegistry, and another subclass is used by CameraGeom
 * for detector-specific coordinate system prefixes (Jim Bosch's clever idea). Thus the shared base class.
 *
 * Comparison is by name, so each unique coordinate system (or prefix) must have a unique name.
 *
 * When switching to unordered_map, a good way to compute the hash is:
 *   size_t hash = 0;
 *   boost::hash_combine(hash, cameraSys.getSysName());
 *   boost::hash_combine(hash, cameraSys.getDetectorName());
 *   return hash;
 */
class CameraSys : public BaseCameraSys {
public:
    /**
     * Construct a CameraSys
     */
    explicit CameraSys(
        std::string const &sysName,         ///< coordinate system name
        std::string const &detectorName=""  /// detector name
    ) : BaseCameraSys(sysName), _detectorName(detectorName) {};

    /// default constructor so SWIG can wrap a vector of pairs containing these
    CameraSys() : BaseCameraSys("?"), _detectorName() {};

    ~CameraSys() {}

    /**
     * Get detector name, or "" if not a detector-specific coordinate system
     */
    std::string getDetectorName() const { return _detectorName; };

    /**
     * Does this have a non-blank detector name?
     */
    bool hasDetectorName() const { return !_detectorName.empty(); }

    bool operator==(CameraSys const &rhs) const {
        return this->getSysName() == rhs.getSysName() && _detectorName == rhs.getDetectorName();
    }

    bool operator!=(CameraSys const &rhs) const {
        return !(*this == rhs);
    }

    // less-than operator required for use in std::map
    bool operator<(CameraSys const &rhs) const {
        if (this->getSysName() == rhs.getSysName()) {
            return _detectorName < rhs.getDetectorName();
        } else {
            return this->getSysName() < rhs.getSysName();
        }
    }

private:
    std::string _detectorName;  ///< detector name; "" if not a detector-specific coordinate system
};

// CameraSys is intended as a key for geom::TransformRegistry, so define these useful types
typedef geom::TransformRegistry<CameraSys> CameraTransformRegistry;
typedef geom::TransformRegistry<CameraSys>::TransformMap CameraTransformMap;

// *** Standard camera coordinate systems ***

/**
 * Focal plane coordinates:
 * Rectilinear x, y (and z when talking about the location of a detector) on the camera focal plane (mm).
 * For z=0 choose a convenient point near the focus at x, y = 0.
 */
extern CameraSys const FOCAL_PLANE;

/**
 * Pupil coordinates:
 * Angular x,y offset from the vertex at the pupil (arcsec).
 */
extern CameraSys const PUPIL;

/**
 * Nominal pixels on the detector (unbinned)
 * This ignores manufacturing imperfections, "tree ring" distortions and all other such effects.
 * It is a uniform grid of rectangular (usually square) pixels.
 *
 * This is a detector prefix; call Detector.getCameraSys(PIXELS) to make a full coordsys.
 */
extern BaseCameraSys const PIXELS;

/**
 * The actual pixels where the photon lands and electrons are generated (unbinned)
 * This takes into account manufacturing defects, "tree ring" distortions and other such effects.
 *
 * This is a detector prefix; call Detector.getCameraSys(ACTUAL_PIXELS) to make a full coordsys.
 */
extern BaseCameraSys const ACTUAL_PIXELS;

std::ostream &operator<< (std::ostream &os, BaseCameraSys const &detSysPrefix);

std::ostream &operator<< (std::ostream &os, CameraSys const &cameraSys);

}}}

#endif
