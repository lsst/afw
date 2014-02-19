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
// this file is meant to be included by TransformMap.h
#include <sstream>
#include <utility>
#include "boost/make_shared.hpp"
#include "lsst/pex/exceptions.h"

namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace geom {

template<typename CoordSys>
TransformMap<CoordSys>::TransformMap(
    CoordSys const &nativeCoordSys,
    Transforms const &transforms
) :
    _nativeCoordSys(nativeCoordSys), _transforms()
{
    for (typename Transforms::const_iterator trIter = transforms.begin();
        trIter != transforms.end(); ++trIter) {
        if (_transforms.count(trIter->first) > 0) {
            std::ostringstream os;
            os << "Duplicate coordSys \"" << trIter->first << "\"";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        } else if (trIter->first == _nativeCoordSys) {
            std::ostringstream os;
            os << "coordSys \"" << trIter->first << "\" matches nativeCoordSys";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        }
        _transforms.insert(*trIter);
    }

    // insert identity transform for nativeCoordSys, if not already provided
    if (!contains(nativeCoordSys)) {
        _transforms.insert(std::make_pair(nativeCoordSys,
            boost::make_shared<IdentityXYTransform>()));
    }
}

template<typename CoordSys>
TransformMap<CoordSys>::TransformMap() : _nativeCoordSys(), _transforms() {}


template<typename CoordSys>
Point2D TransformMap<CoordSys>::transform(
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
std::vector<Point2D> TransformMap<CoordSys>::transform(
    std::vector<Point2D> const &pointList,
    CoordSys const &fromCoordSys,
    CoordSys const &toCoordSys
) const {
    if (fromCoordSys == toCoordSys) {
        return pointList;
    }

    std::vector<Point2D> outList;

    // transform pointList from fromCoordSys to native coords, filling outList
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

    // transform outList from native coords to toCoordSys, in place
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
std::vector<CoordSys> TransformMap<CoordSys>::getCoordSysList() const {
    std::vector<CoordSys> coordSysList;
    for (typename Transforms::const_iterator trIter = _transforms.begin();
        trIter != _transforms.end(); ++trIter) {
        coordSysList.push_back(trIter->first);
    }
    return coordSysList;
}

template<typename CoordSys>
CONST_PTR(XYTransform) TransformMap<CoordSys>::operator[](
    CoordSys const &coordSys
) const {
    typename Transforms::const_iterator const foundIter = _transforms.find(coordSys);
    if (foundIter == _transforms.end()) {
        std::ostringstream os;
        os << "Registry does not support coordSys \"" << coordSys << "\"";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    return foundIter->second;
}

template<typename CoordSys>
bool TransformMap<CoordSys>::contains(
    CoordSys const &coordSys
) const {
    return _transforms.find(coordSys) != _transforms.end();
}

}}}
