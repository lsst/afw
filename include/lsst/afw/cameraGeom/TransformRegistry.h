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
 
#if !defined(LSST_AFW_CAMERAGEOM_TRANSFORMREGISTRY_H)
#define LSST_AFW_CAMERAGEOM_TRANSFORMREGISTRY_H

#include <string>
#include <map>
#include <utility>
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/XYTransform.h"
#include "lsst/afw/cameraGeom/CameraPoint.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * A registry of 2-dimensional coordinate transforms
 */
class TransformRegistry {
public:
    /**
     * Construct a TransformRegistry
     *
     * @raise pexExcept::InvalidParameterException if you specify the same coordSys
     * more than once, or a transform is specified where coordSys == nativeCoordSys
     */
    explicit TransformRegistry(
        std::string const &nativeCoordSys,   ///< Native coordinate system for this registry;
            ///< all XYTransforms in the registry must convert to this coordinate system
        std::vector<std::pair<std::string, CONST_PTR(geom::XYTransform)> > const &transformRegistry
            ///< xy transforms: a list of pairs of:
            ///< * coordSys: coordinate system name
            ///< * xyTransform: an XYTransform that converts from coordSys to nativeCoordSys (and back again)
    );

    /**
     * Convert a point from one coordinate system to another
     *
     * @raise: pexExcept::InvalidParameterException if toCoordSys is unknown
     */
    CameraPoint convert(
        CameraPoint const &cameraPoint, ///< point from which to convert
        std::string const &toCoordSys   ///< coordinate system to which to convert
    ) const;

    /**
     * Convert a list of Point2D from one coordinate system to another
     *
     * @raise: pexExcept::InvalidParameterException if fromCoordSys or toCoordSys is unknown
     */
     std::vector<geom::Point2D> convert(
        std::vector<geom::Point2D> const &pointList,    ///< list of points to convert
        std::string const &fromCoordSys,    ///< from coordinate system
        std::string const &toCoordSys       ///< to coordinate system
    ) const;

    std::string getNativeCoordSys() const { return _nativeCoordSys; }

    /**
     * Get an XYTransform that converts from coordSys to nativeCoordSys
     *
     * @return XYTransform that converts from coordSys to nativeCoordSys
     * (if coordSys==nativeCoordSys then returns IdentityXYTransform).
     *
     * @raise: pexExcept::InvalidParameterException if coordSys is unknown
     */
    CONST_PTR(geom::XYTransform) getXYTransform(
        std::string const &coordSys ///< coordinate system whose XYTransform is wanted;
            ///< must not be the native coordinate system
    ) const;

    /**
     * Get a list of supported coordinate systems
     *
     * @return a list of coordinate systems, starting with nativeCoordSys
     */
    std::vector<std::string> getCoordSysList() const;

 
private:
    typedef std::map<std::string, CONST_PTR(geom::XYTransform)>::const_iterator _MapIter;
    std::string _nativeCoordSys;        ///< native coordinate system
    std::map<std::string, CONST_PTR(geom::XYTransform)> _transformMap; 
        ///< map of coordSys: XYTransform for coordSys->nativeCoordSys and back again
};

}}}

#endif
