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
 
#if !defined(LSST_AFW_GEOM_TRANSFORMREGISTRY_H)
#define LSST_AFW_GEOM_TRANSFORMREGISTRY_H

#include <string>
#include <utility>
#include <vector>
#include <map>
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/XYTransform.h"

namespace lsst {
namespace afw {
namespace geom {

/**
 * A registry of 2-dimensional coordinate transforms
 *
 * If CoordSys is not a plain old data type or std::string then:
 * * CoordSys must have a default constructor (no arguments) to allow SWIG to wrap a collection
 * * CoordSys must support operator< to support use as a key in std::map
 * * CoordSys should support operator== and operator!= for common sense
 * * CoordSys should support __hash__ in Python to support proper behavior in sets and dicts
 * * You must overload ostream operator<<(CoordSys const &) to support error messages in TransformRegistry
 * For an example see ../cameraGeom/CameraSys.h (CameraSys is used as a CoordSys) and its SWIG wrapper.
 *
 * At some point we will switch to using std::unordered_map (once we switch to C++11 and a SWIG that supports
 * its collection classes). At that point instead of requiring operator<, it will be necessary to
 * specialize std::hash<CoordSys>(CoordSys const &).
 */
template<typename CoordSys>
class TransformRegistry {
public:
    typedef std::map<CoordSys, CONST_PTR(XYTransform)> TransformMap;
    // the following is needed by SWIG; see TransformRegistry.i
    typedef CoordSys _CoordSysType;

    /**
     * Construct a TransformRegistry
     *
     * @note If transformMap includes a transform for nativeCoordSys
     * then it is used (without checking); if not, then a unity transform is added.
     *
     * @throw pexExcept::InvalidParameterException if you specify the same coordSys
     * more than once, or a transform is specified where coordSys == nativeCoordSys
     */
    explicit TransformRegistry(
        CoordSys const &nativeCoordSys,   ///< Native coordinate system for this registry
        TransformMap const &transformMap ///< xy transforms: a map of coordSys:xyTransform,
            ///< where xyTransform.forward converts coordSys to nativeCoordSys
    );

    /// null implementation to make SWIG willing to wrap a vector that contains these
    explicit TransformRegistry();

    ~TransformRegistry() {}

    /**
     * Convert a point from one coordinate system to another
     *
     * @return the converted value as a Point2D
     *
     * @throw pexExcept::InvalidParameterException if toCoordSys is unknown
     */
    Point2D convert(
        Point2D const &fromPoint,       ///< point from which to convert
        CoordSys const &fromSys,        ///< coordinate system from which to convert
        CoordSys const &toCoordSys      ///< coordinate system to which to convert
    ) const;

    /**
     * Convert a list of Point2D from one coordinate system to another
     *
     * @throw pexExcept::InvalidParameterException if fromCoordSys or toCoordSys is unknown
     */
     std::vector<Point2D> convert(
        std::vector<Point2D> const &pointList,    ///< list of points to convert
        CoordSys const &fromCoordSys,    ///< from coordinate system
        CoordSys const &toCoordSys       ///< to coordinate system
    ) const;

    CoordSys getNativeCoordSys() const { return _nativeCoordSys; }

    /**
     * Get a list of supported coordinate systems
     *
     * @return a list of coordinate systems, in undefined order.
     */
    std::vector<CoordSys> getCoordSysList() const;

    /**
     * Return true if the coordinate system is supported
     *
     * In Python this is renamed to __contains__; use as follows:
     *     coordSys in transformRegistry
     */
    bool contains(
        CoordSys const &coordSys ///< coordinate system
    ) const;

    /**
     * Get an XYTransform that converts from coordSys to nativeCoordSys in the forward direction
     *
     * @return an XYTransform
     *
     * @throw pexExcept::InvalidParameterException if coordSys is unknown
     */
    CONST_PTR(XYTransform) operator[](
        CoordSys const &coordSys ///< coordinate system whose XYTransform is wanted
    ) const;

    typename TransformMap::const_iterator begin() const { return _transformMap.begin(); }

    typename TransformMap::const_iterator end() const { return _transformMap.end(); }

    size_t size() const { return _transformMap.size(); }

private:
    CoordSys _nativeCoordSys;   ///< native coordinate system
    TransformMap _transformMap; ///< map of coordSys: XYTransform
};

}}}

/// the implementation code must be included so other users can make templated versions
#include "lsst/afw/geom/TransformRegistry.cc"

#endif
