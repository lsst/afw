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
// I would rather include <tr1/unordered_map> but I get symbol collisions from the utils package
#include "lsst/tr1/unordered_map.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/XYTransform.h"

namespace lsst {
namespace afw {
namespace geom {

/**
 * A registry of 2-dimensional coordinate transforms
 *
 * If CoordSys is not a plain old data type then you must define functor boost::hash<CoordSys>;
 * see ../geom/CameraSys.h for an example.
 */
template<typename CoordSys>
class TransformRegistry {
public:
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
        std::vector<std::pair<CoordSys, CONST_PTR(XYTransform)> > const &transformRegistry
            ///< xy transforms: a list of pairs of:
            ///< * coordSys: coordinate system
            ///< * xyTransform: an XYTransform whose forward method converts coordSys->nativeCoordSys
    );

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
     * Get an XYTransform that converts from coordSys to nativeCoordSys in the forward direction
     *
     * @return an XYTransform
     *
     * @throw pexExcept::InvalidParameterException if coordSys is unknown
     */
    CONST_PTR(XYTransform) getXYTransform(
        CoordSys const &coordSys ///< coordinate system whose XYTransform is wanted
    ) const;

    /**
     * Return true if the coordinate system is supported
     */
    bool hasXYTransform(
        CoordSys const &coordSys ///< coordinate system
    ) const;

 
private:
    typedef std::tr1::unordered_map<CoordSys, CONST_PTR(XYTransform)> _MapType;
    CoordSys _nativeCoordSys;   ///< native coordinate system
    _MapType _transformMap;     ///< map of coordSys: XYTransform
};

}}}

#endif
