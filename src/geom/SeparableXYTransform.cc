// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include <cmath>

#include "lsst/afw/geom/Functor.h"
#include "lsst/afw/geom/SeparableXYTransform.h"

namespace lsst {
namespace afw {
namespace geom {

SeparableXYTransform::
SeparableXYTransform(Functor const & xfunctor, Functor const & yfunctor)
   : XYTransform(), _xfunctor(xfunctor.clone()), _yfunctor(yfunctor.clone()) {
}

PTR(XYTransform) SeparableXYTransform::clone() const {
   return boost::make_shared<SeparableXYTransform>(*_xfunctor, *_yfunctor);
}

Point2D SeparableXYTransform::forwardTransform(Point2D const & point) const {
   double xin = point.getX();
   double yin = point.getY();
   double xout = (*_xfunctor)(xin);
   double yout = (*_yfunctor)(yin);
   return Point2D(xout, yout);
}

Point2D SeparableXYTransform::reverseTransform(Point2D const & point) const {
   double xout = point.getX();
   double yout = point.getY();
   double xin = 0;
   double yin = 0;
   try {
      xin = _xfunctor->inverse(xout);
      yin = _yfunctor->inverse(yout);
   } catch (lsst::pex::exceptions::Exception & e) {
      LSST_EXCEPT_ADD(e, "Called from SeparableXYTransform::reverseTransform");
      throw;
   }
   return Point2D(xin, yin);
}

Functor const & SeparableXYTransform::getXfunctor() const {
   return *_xfunctor;
}

Functor const & SeparableXYTransform::getYfunctor() const {
   return *_yfunctor;
}

} // namespace geom
} // namespace afw
} // namespace lsst
