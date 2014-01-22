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
#include <sstream>
#include "boost/make_shared.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/cameraGeom/TransformRegistry.h"

namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace cameraGeom {

TransformRegistry::TransformRegistry(
    std::string const &nativeCoordSys,
    std::vector<std::pair<std::string, CONST_PTR(geom::XYTransform)> > const &transformRegistry
) :
    _nativeCoordSys(nativeCoordSys)
{
    typedef std::vector<std::pair<std::string, CONST_PTR(geom::XYTransform)> >::const_iterator ListIter;
    for (ListIter trIter = transformRegistry.begin(); trIter != transformRegistry.end(); ++trIter) {
        if (_transformMap.count(trIter->first) > 0) {
            std::ostringstream os;
            os << "Duplicate coordSys \"" << trIter->first << "\"";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        } else if (trIter->first == _nativeCoordSys) {
            std::ostringstream os;
            os << "coordSys \"" << trIter->first << "\" matches nativeCoordSys";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        }
        _transformMap.insert(*trIter);
    }
}

CameraPoint TransformRegistry::convert(
    CameraPoint const &fromPoint,
    std::string const &toCoordSys
) const {
    std::string fromCoordSys = fromPoint.getCoordSys();
    if (fromCoordSys == toCoordSys) {
        return fromPoint;
    }

    geom::Point2D outPoint2D;

    // compute outPoint2D = fromPoint converted to native coords
    if (fromCoordSys != _nativeCoordSys) {
        CONST_PTR(geom::XYTransform) fromTransform = getXYTransform(fromCoordSys);
        outPoint2D = fromTransform->forwardTransform(fromPoint.getPoint());
    } else {
        outPoint2D = fromPoint.getPoint();
    }

    // convert outPoint2D from native coords to toCoordSys
    if (toCoordSys != _nativeCoordSys) {
        CONST_PTR(geom::XYTransform) toTransform = getXYTransform(toCoordSys);
        outPoint2D = toTransform->reverseTransform(outPoint2D);
    }
    return CameraPoint(outPoint2D, toCoordSys, fromPoint.getFrameName());
}

std::vector<geom::Point2D> TransformRegistry::convert(
    std::vector<geom::Point2D> const &pointList,
    std::string const &fromCoordSys,
    std::string const &toCoordSys
) const {
    if (fromCoordSys == toCoordSys) {
        return pointList;
    }

    std::vector<geom::Point2D> outList;

    // convert pointList from fromCoordSys to native coords, filling outList
    if (fromCoordSys != _nativeCoordSys) {
        CONST_PTR(geom::XYTransform) fromTransform = getXYTransform(fromCoordSys);
        for (std::vector<geom::Point2D>::const_iterator fromPtIter = pointList.begin();
            fromPtIter != pointList.end(); ++fromPtIter) {
            outList.push_back(fromTransform->forwardTransform(*fromPtIter));
        }
    } else {
        for (std::vector<geom::Point2D>::const_iterator fromPtIter = pointList.begin();
            fromPtIter != pointList.end(); ++fromPtIter) {
            outList.push_back(*fromPtIter);
        }
    }

    // convert outList from native coords to toCoordSys, in place
    if (toCoordSys != _nativeCoordSys) {
        CONST_PTR(geom::XYTransform) toTransform = getXYTransform(toCoordSys);
        for (std::vector<geom::Point2D>::iterator nativePtIter = outList.begin();
            nativePtIter != pointList.end(); ++nativePtIter) {
            *nativePtIter = toTransform->reverseTransform(*nativePtIter);
        }
    }
    return outList;
}


CONST_PTR(geom::XYTransform) TransformRegistry::getXYTransform(
    std::string const &coordSys
) const {
    if (coordSys == _nativeCoordSys) {
        return boost::make_shared<geom::IdentityXYTransform>(false);
    }

    _MapIter const foundIter = _transformMap.find(coordSys);
    if (foundIter == _transformMap.end()) {
        std::ostringstream os;
        os << "Registry does not support coordSys \"" << coordSys << "\"";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    return foundIter->second;
}

std::vector<std::string> TransformRegistry::getCoordSysList() const {
    std::vector<std::string> coordSysList;
    coordSysList.push_back(_nativeCoordSys);
    for (_MapIter trIter = _transformMap.begin(); trIter != _transformMap.end(); ++trIter) {
        coordSysList.push_back(trIter->first);
    }
    return coordSysList;
}

}}}
