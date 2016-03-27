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

XYTransform::XYTransform() 
    : daf::base::Citizen(typeid(this))
{ }


/// default implementation; subclass may override
AffineTransform XYTransform::linearizeForwardTransform(Point2D const &p) const
{
    Point2D px = p + Extent2D(1,0);
    Point2D py = p + Extent2D(0,1);

    return makeAffineTransformFromTriple(p, px, py, 
                                                  this->forwardTransform(p),
                                                  this->forwardTransform(px), 
                                                  this->forwardTransform(py));
}


/// default implementation; subclass may override
AffineTransform XYTransform::linearizeReverseTransform(Point2D const &p) const
{
    Point2D px = p + Extent2D(1,0);
    Point2D py = p + Extent2D(0,1);

    return makeAffineTransformFromTriple(p, px, py, 
                                                  this->reverseTransform(p),
                                                  this->reverseTransform(px), 
                                                  this->reverseTransform(py));
}


/// default implementation; subclass may override
PTR(XYTransform) XYTransform::invert() const
{
    return boost::make_shared<InvertedXYTransform> (this->clone());
}

}}}
