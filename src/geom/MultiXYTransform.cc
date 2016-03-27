// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/XYTransform.h"
#include "boost/make_shared.hpp"

namespace pexEx = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace geom {

MultiXYTransform::MultiXYTransform(std::vector<CONST_PTR(XYTransform)> const &transformList)
    : XYTransform(), _transformList(transformList)
{
    for (TransformList::const_iterator trIter = _transformList.begin();
        trIter != _transformList.end(); ++trIter) {
        if (!bool(*trIter)) {
            throw LSST_EXCEPT(pexEx::InvalidParameterError, "One or more transforms is null");
        }
    }
}

PTR(XYTransform) MultiXYTransform::clone() const
{
    return boost::make_shared<MultiXYTransform>(_transformList);
}

Point2D MultiXYTransform::forwardTransform(Point2D const &point) const
{
    Point2D retPoint = point;
    for (TransformList::const_iterator trIter = _transformList.begin();
        trIter != _transformList.end(); ++trIter) {
        retPoint = (*trIter)->forwardTransform(retPoint);
    }
    return retPoint;
}

Point2D MultiXYTransform::reverseTransform(Point2D const &point) const
{
    Point2D retPoint = point;
    for (TransformList::const_reverse_iterator trIter = _transformList.rbegin();
        trIter != _transformList.rend(); ++trIter) {
        retPoint = (*trIter)->reverseTransform(retPoint);
    }
    return retPoint;
}

AffineTransform MultiXYTransform::linearizeForwardTransform(Point2D const &point) const
{
    Point2D tempPt = point;
    AffineTransform retTransform = AffineTransform();
    for (TransformList::const_iterator trIter = _transformList.begin();
        trIter != _transformList.end(); ++trIter) {
        AffineTransform tempTransform = (*trIter)->linearizeForwardTransform(tempPt);
        tempPt = tempTransform(tempPt);
        retTransform = tempTransform * retTransform;
    }
    return retTransform;
}

AffineTransform MultiXYTransform::linearizeReverseTransform(Point2D const &point) const
{
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

}}}
