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
// I would rather include <tr1/unordered_map> but I get symbol collisions from the utils package,
// due to lsst/tr1/unordered_map.h, which pretends that boost::unordered_map is std::tr1::unordered_map;
// this pretence falls apart if one specializes the hash function, since boost and tr1 do that differently
#include "boost/unordered_map.hpp"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/XYTransform.h"

namespace lsst {
namespace afw {
namespace geom {

/**
 * A registry of 2-dimensional coordinate transforms
 *
 * If CoordSys is not a plain old data type or std::string then:
 * * CoordSys must have a default constructor (no arguments)
 * * CoordSys must have member function operator==
 * * You must define function hash_value(CoordSys const &)
 * * You must overload ostream operator<<(CoordSys const &)
 * For an example see ../cameraGeom/CameraSys.h (CameraSys is used as a CoordSys)
 * The reason for these rules is to allow CoordSys to be used as a key in boost::unordered_map,
 * and to allow SWIG to wrap a collection containing CoordSys.
 *
 * @warning When we switch to using std::unordered_map then you must define functor
 * std::hash<CoordSys>(CoordSys const &) instead of function hash_value.
 */
template<typename CoordSys>
class TransformRegistry {
public:
    typedef std::vector<std::pair<CoordSys, CONST_PTR(XYTransform)> > TransformList;
    typedef boost::unordered_map<CoordSys, CONST_PTR(XYTransform)> TransformMap;

    /**
     * Construct a TransformRegistry
     *
     * @note If transformRegistry includes a transform for nativeCoordSys
     * then it is used (without checking); if not, then a unity transform is added.
     *
     * @throw pexExcept::InvalidParameterException if you specify the same coordSys
     * more than once, or a transform is specified where coordSys == nativeCoordSys
     */
    explicit TransformRegistry(
        CoordSys const &nativeCoordSys,   ///< Native coordinate system for this registry;
            ///< all XYTransforms in the registry must convert to this coordinate system
        TransformList const &transformRegistry
            ///< xy transforms: a list of pairs of:
            ///< * coordSys: coordinate system
            ///< * xyTransform: an XYTransform whose forward method converts coordSys->nativeCoordSys
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
     */
    bool hasTransform(
        CoordSys const &coordSys ///< coordinate system
    ) const;

    /**
     * Get an XYTransform that converts from coordSys to nativeCoordSys in the forward direction
     *
     * @return an XYTransform
     *
     * @throw pexExcept::InvalidParameterException if coordSys is unknown
     */
    CONST_PTR(XYTransform) getTransform(
        CoordSys const &coordSys ///< coordinate system whose XYTransform is wanted
    ) const;

    typename TransformMap::const_iterator begin() const { return _transformMap.begin(); }

    typename TransformMap::const_iterator end() const { return _transformMap.end(); }

    size_t size() const { return _transformMap.size(); }

    /**
     * Return a list of transforms
     */
    TransformList getTransformList() const;

private:
    CoordSys _nativeCoordSys;   ///< native coordinate system
    TransformMap _transformMap; ///< map of coordSys: XYTransform
};

}}}

/// the implementation code must be included so other users can make templated versions
#include "lsst/afw/geom/TransformRegistry.cc"

#endif
