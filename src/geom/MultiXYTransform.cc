// -*- lsst-c++ -*-

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

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/XYTransform.h"
#include <memory>

namespace pexEx = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace geom {

MultiXYTransform::MultiXYTransform(std::vector<std::shared_ptr<XYTransform const>> const &transformList)
        : XYTransform(), _transformList(transformList) {
    for (TransformList::const_iterator trIter = _transformList.begin(); trIter != _transformList.end();
         ++trIter) {
        if (!bool(*trIter)) {
            throw LSST_EXCEPT(pexEx::InvalidParameterError, "One or more transforms is null");
        }
    }
}

MultiXYTransform::MultiXYTransform(MultiXYTransform const &) = default;
MultiXYTransform::MultiXYTransform(MultiXYTransform &&) = default;
MultiXYTransform &MultiXYTransform::operator=(MultiXYTransform const &) = default;
MultiXYTransform &MultiXYTransform::operator=(MultiXYTransform &&) = default;
MultiXYTransform::~MultiXYTransform() = default;

std::shared_ptr<XYTransform> MultiXYTransform::clone() const {
    return std::make_shared<MultiXYTransform>(_transformList);
}

Point2D MultiXYTransform::forwardTransform(Point2D const &point) const {
    Point2D retPoint = point;
    for (TransformList::const_iterator trIter = _transformList.begin(); trIter != _transformList.end();
         ++trIter) {
        retPoint = (*trIter)->forwardTransform(retPoint);
    }
    return retPoint;
}

Point2D MultiXYTransform::reverseTransform(Point2D const &point) const {
    Point2D retPoint = point;
    for (TransformList::const_reverse_iterator trIter = _transformList.rbegin();
         trIter != _transformList.rend(); ++trIter) {
        retPoint = (*trIter)->reverseTransform(retPoint);
    }
    return retPoint;
}

AffineTransform MultiXYTransform::linearizeForwardTransform(Point2D const &point) const {
    Point2D tempPt = point;
    AffineTransform retTransform = AffineTransform();
    for (TransformList::const_iterator trIter = _transformList.begin(); trIter != _transformList.end();
         ++trIter) {
        AffineTransform tempTransform = (*trIter)->linearizeForwardTransform(tempPt);
        tempPt = tempTransform(tempPt);
        retTransform = tempTransform * retTransform;
    }
    return retTransform;
}

AffineTransform MultiXYTransform::linearizeReverseTransform(Point2D const &point) const {
    Point2D tempPt = point;
    AffineTransform retTransform = AffineTransform();
    for (TransformList::const_reverse_iterator trIter = _transformList.rbegin();
         trIter != _transformList.rend(); ++trIter) {
        AffineTransform tempTransform = (*trIter)->linearizeReverseTransform(tempPt);
        tempPt = tempTransform(tempPt);
        retTransform = tempTransform * retTransform;
    }
    return retTransform;
}
}  // namespace geom
}  // namespace afw
}  // namespace lsst
