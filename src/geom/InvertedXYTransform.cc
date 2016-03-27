// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "lsst/afw/geom/XYTransform.h"
#include "boost/make_shared.hpp"

namespace lsst {
namespace afw {
namespace geom {

InvertedXYTransform::InvertedXYTransform(CONST_PTR(XYTransform) base)
    : XYTransform(), _base(base)
{ }

PTR(XYTransform) InvertedXYTransform::clone() const
{
    // deep copy
    return boost::make_shared<InvertedXYTransform> (_base->clone());
}

PTR(XYTransform) InvertedXYTransform::invert() const
{
    return _base->clone();
}

Point2D InvertedXYTransform::forwardTransform(Point2D const &point) const
{
    return _base->reverseTransform(point);
}

Point2D InvertedXYTransform::reverseTransform(Point2D const &point) const
{
    return _base->forwardTransform(point);
}

AffineTransform InvertedXYTransform::linearizeForwardTransform(Point2D const &point) const
{
    return _base->linearizeReverseTransform(point);
}

AffineTransform InvertedXYTransform::linearizeReverseTransform(Point2D const &point) const
{
    return _base->linearizeForwardTransform(point);
}

}}}
