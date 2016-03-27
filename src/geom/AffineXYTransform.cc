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

AffineXYTransform::AffineXYTransform(AffineTransform const &affineTransform)
    : XYTransform(), _forwardAffineTransform(affineTransform), 
      _reverseAffineTransform(_forwardAffineTransform.invert())
{ }

PTR(XYTransform) AffineXYTransform::clone() const
{
    return boost::make_shared<AffineXYTransform> (_forwardAffineTransform);
}

Point2D AffineXYTransform::forwardTransform(Point2D const &position) const
{
    return _forwardAffineTransform(position);
}

Point2D AffineXYTransform::reverseTransform(Point2D const &position) const
{
    return _reverseAffineTransform(position);
}

AffineTransform AffineXYTransform::linearizeForwardTransform(Point2D const &) const
{
    return _forwardAffineTransform;
}

AffineTransform AffineXYTransform::linearizeReverseTransform(Point2D const &) const
{
    return _reverseAffineTransform; 
}

AffineTransform AffineXYTransform::getForwardTransform() const
{
    return _forwardAffineTransform;
}

AffineTransform AffineXYTransform::getReverseTransform() const
{
    return _reverseAffineTransform;
}

}}}
