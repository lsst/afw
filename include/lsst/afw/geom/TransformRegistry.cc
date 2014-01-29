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
#include <utility>
#include "boost/make_shared.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/TransformRegistry.h"

namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace geom {

template<typename CoordSys>
TransformRegistry<CoordSys>::TransformRegistry(
    CoordSys const &nativeCoordSys,
    TransformMap const &transformList
) :
    _nativeCoordSys(nativeCoordSys), _transformMap()
{
    for (typename TransformMap::const_iterator trIter = transformList.begin();
        trIter != transformList.end(); ++trIter) {
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

    // insert identity transform for nativeCoordSys, if not already provided
    if (!contains(nativeCoordSys)) {
        _transformMap.insert(std::make_pair(nativeCoordSys,
            boost::make_shared<IdentityXYTransform>(false)));
    }
}

template<typename CoordSys>
TransformRegistry<CoordSys>::TransformRegistry() : _nativeCoordSys(), _transformMap() {}


template<typename CoordSys>
Point2D TransformRegistry<CoordSys>::convert(
    Point2D const &fromPoint,
    CoordSys const &fromCoordSys,
    CoordSys const &toCoordSys
) const {
    if (fromCoordSys == toCoordSys) {
        return fromPoint;
    }

    // transform fromSys -> nativeSys -> toSys
    CONST_PTR(XYTransform) fromTransform = (*this)[fromCoordSys];
    CONST_PTR(XYTransform) toTransform = (*this)[toCoordSys];
    return toTransform->reverseTransform(fromTransform->forwardTransform(fromPoint));
}

template<typename CoordSys>
std::vector<Point2D> TransformRegistry<CoordSys>::convert(
    std::vector<Point2D> const &pointList,
    CoordSys const &fromCoordSys,
    CoordSys const &toCoordSys
) const {
    if (fromCoordSys == toCoordSys) {
        return pointList;
    }

    std::vector<Point2D> outList;

    // convert pointList from fromCoordSys to native coords, filling outList
    if (fromCoordSys != _nativeCoordSys) {
        CONST_PTR(XYTransform) fromTransform = (*this)[fromCoordSys];
        for (std::vector<Point2D>::const_iterator fromPtIter = pointList.begin();
            fromPtIter != pointList.end(); ++fromPtIter) {
            outList.push_back(fromTransform->forwardTransform(*fromPtIter));
        }
    } else {
        for (std::vector<Point2D>::const_iterator fromPtIter = pointList.begin();
            fromPtIter != pointList.end(); ++fromPtIter) {
            outList.push_back(*fromPtIter);
        }
    }

    // convert outList from native coords to toCoordSys, in place
    if (toCoordSys != _nativeCoordSys) {
        CONST_PTR(XYTransform) toTransform = (*this)[toCoordSys];
        for (std::vector<Point2D>::iterator nativePtIter = outList.begin();
            nativePtIter != pointList.end(); ++nativePtIter) {
            *nativePtIter = toTransform->reverseTransform(*nativePtIter);
        }
    }
    return outList;
}

template<typename CoordSys>
std::vector<CoordSys> TransformRegistry<CoordSys>::getCoordSysList() const {
    std::vector<CoordSys> coordSysList;
    for (typename TransformMap::const_iterator trIter = _transformMap.begin();
        trIter != _transformMap.end(); ++trIter) {
        coordSysList.push_back(trIter->first);
    }
    return coordSysList;
}

template<typename CoordSys>
CONST_PTR(XYTransform) TransformRegistry<CoordSys>::operator[](
    CoordSys const &coordSys
) const {
    typename TransformMap::const_iterator const foundIter = _transformMap.find(coordSys);
    if (foundIter == _transformMap.end()) {
        std::ostringstream os;
        os << "Registry does not support coordSys \"" << coordSys << "\"";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    return foundIter->second;
}

template<typename CoordSys>
bool TransformRegistry<CoordSys>::contains(
    CoordSys const &coordSys
) const {
    return _transformMap.find(coordSys) != _transformMap.end();
}

}}}
