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
 
#if !defined(LSST_AFW_GEOM_COORDSYS_H)
#define LSST_AFW_GEOM_COORDSYS_H

#include <string>
// #include <tr1/functional> // for hash

namespace lsst {
namespace afw {
namespace geom {

/**
 * Base class for coordinate system keys used in in TransformMap
 *
 * @note A subclass is used for keys in TransformMap, and another subclass is used by CameraGeom
 * for detector-specific coordinate system prefixes (Jim Bosch's clever idea). Thus the shared base class.
 *
 * Comparison is by name, so each unique coordinate system (or prefix) must have a unique name.
 */
class BaseCoordSys {
public:
    explicit BaseCoordSys(std::string const &name) : _name(name) {}
    ~BaseCoordSys() {}

    /**
     * Get name as <className>(arg1, arg2...)
     *
     * The name must be unique for a unique coordinate system,
     * since it is used by getHash operator== and other operators.
     */
    std::string getName() const { return _name; };

    // /** get a hash; allows use in std::unordered_map */
    // size_t getHash() const {
    //     return std::tr1::hash<std::string>(_name);
    // }

    /** equals operator; based on name */
    bool operator==(BaseCoordSys const &rhs) const {
        return _name == rhs.getName();
    }

    /** not equals operator; based on name */
    bool operator!=(BaseCoordSys const &rhs) const {
        return _name != rhs.getName();
    }

    /** less-than operator; based on name; allows use in std::map */
    bool operator<(BaseCoordSys const &rhs) const {
        return _name < rhs.getName();
    }
public:
    std::string _name;
};

/**
 * Class used for keys in TransformMap
 *
 * Each coordinate system must have a unique name. Hashing and equality testing are based on this name.
 */
class CoordSys : public BaseCoordSys {
public:
    explicit CoordSys(std::string const &name) : BaseCoordSys(name) {}
    ~CoordSys() {}
};

#endif
