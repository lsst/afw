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
 
#if !defined(LSST_AFW_GEOM_TRANSFORMMAP_H)
#define LSST_AFW_GEOM_TRANSFORMMAP_H

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
 * A registry of 2-dimensional coordinate transforms, templated on a coordinate system type
 *
 * Contains a native CoordSysT and a map of CoordSysT:XYTransform (including an identity transform
 * entry for native CoordSysT). Each map entry is an XYTransform whose forwardTransform method
 * converts from the native system to CoordSysT.
 *
 * TransformMap supports transforming between any two supported CoordSysT using the transform method.
 * It also allows iteration over the map of CoordSysT: XYTransform:
 * * In C++ the iterator is a CoordSysT, CONST_PTR(XYTransform) pair.
 * * In Python, the iterator returns a CoordSysT; use TransformMap[CoordSysT] to access the XYTransform.
 *
 * If CoordSysT is not a plain old data type or std::string then:
 * * CoordSysT must have a default constructor (no arguments), so SWIG can wrap some collections
 * * CoordSysT must support operator< to support use as a key in std::map
 * * CoordSysT should support operator== and operator!= for common sense
 * * CoordSysT should support __hash__ in Python to support proper behavior in sets and dicts
 * * You must overload ostream operator<<(CoordSysT const &) to support error messages in TransformMap
 * For an example see ../cameraGeom/CameraSys.h (CameraSys is used as a CoordSysT) and its SWIG wrapper.
 *
 * At some point we will switch to using std::unordered_map (once we switch to C++11 and a SWIG that supports
 * its collection classes). At that point instead of requiring CoordSysT.operator<, it will be necessary to
 * specialize std::hash<CoordSysT>(CoordSysT const &).
 *
 * @warning: code that instantiates a templated version of TransformMap must include
 * lsst/afw/geom/TransformMapImpl.h (else you will get linker errors).
 */
template<typename CoordSysT>
class TransformMap {
public:
    typedef std::map<CoordSysT, CONST_PTR(XYTransform)> Transforms;
    typedef CoordSysT CoordSys; // needed by SWIG; see TransformMap.i

    /**
     * Construct a TransformMap
     *
     * @note If transformMap includes a transform for nativeCoordSys
     * then it is used (without checking); if not, then a unity transform is added.
     *
     * @throw pexExcept::InvalidParameterException if you specify the same coordSys
     * more than once, or a transform is specified where coordSys == nativeCoordSys
     */
    explicit TransformMap(
        CoordSysT const &nativeCoordSys, ///< Native coordinate system for this registry
        Transforms const &transforms    ///< a map of coordSys:xyTransform,
            ///< where xyTransform.forward transforms coordSys to nativeCoordSys
    );

    /// null implementation to make SWIG willing to wrap a map that contains these
    explicit TransformMap();

    /**
     * Convert a point from one coordinate system to another
     *
     * @return the transformed value as a Point2D
     *
     * @throw pexExcept::InvalidParameterException if toCoordSys is unknown
     */
    Point2D transform(
        Point2D const &fromPoint,       ///< point from which to transform
        CoordSysT const &fromSys,        ///< coordinate system from which to transform
        CoordSysT const &toCoordSys      ///< coordinate system to which to transform
    ) const;

    /**
     * Convert a list of Point2D from one coordinate system to another
     *
     * @throw pexExcept::InvalidParameterException if fromCoordSys or toCoordSys is unknown
     */
     std::vector<Point2D> transform(
        std::vector<Point2D> const &pointList,    ///< list of points to transform
        CoordSysT const &fromCoordSys,    ///< from coordinate system
        CoordSysT const &toCoordSys       ///< to coordinate system
    ) const;

    CoordSysT getNativeCoordSys() const { return _nativeCoordSys; }

    /**
     * Get a list of supported coordinate systems
     *
     * @return a list of coordinate systems, in undefined order.
     */
    std::vector<CoordSysT> getCoordSysList() const;

    /**
     * Return true if the coordinate system is supported
     *
     * In Python this is renamed to __contains__; use as follows:
     *     coordSys in transformMap
     */
    bool contains(
        CoordSysT const &coordSys ///< coordinate system
    ) const;

    /**
     * Get an XYTransform that transforms from coordSys to nativeCoordSys in the forward direction
     *
     * @return an XYTransform
     *
     * @throw pexExcept::InvalidParameterException if coordSys is unknown
     */
    CONST_PTR(XYTransform) operator[](
        CoordSysT const &coordSys ///< coordinate system whose XYTransform is wanted
    ) const;

    typename Transforms::const_iterator begin() const { return _transforms.begin(); }

    typename Transforms::const_iterator end() const { return _transforms.end(); }

    size_t size() const { return _transforms.size(); }

private:
    CoordSysT _nativeCoordSys;   ///< native coordinate system
    Transforms _transforms;   ///< map of coordSys: XYTransform
};

}}}

#endif
