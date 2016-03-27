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

IdentityXYTransform::IdentityXYTransform()
    : XYTransform()
{ }

PTR(XYTransform) IdentityXYTransform::clone() const
{
    return boost::make_shared<IdentityXYTransform> ();
}

Point2D IdentityXYTransform::forwardTransform(Point2D const &point) const
{
    return point;
}

Point2D IdentityXYTransform::reverseTransform(Point2D const &point) const
{
    return point;
}

AffineTransform IdentityXYTransform::linearizeForwardTransform(Point2D const &point) const
{
    // note: AffineTransform constructor called with no arguments gives the identity transform
    return AffineTransform(); 
}

AffineTransform IdentityXYTransform::linearizeReverseTransform(Point2D const &point) const
{
    // note: AffineTransform constructor called with no arguments gives the identity transform
    return AffineTransform(); 
}

}}}
