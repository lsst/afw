// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "boost/make_shared.hpp"
#include "lsst/afw/geom/Functor.h"

namespace lsst {
namespace afw {
namespace geom {

LinearFunctor::LinearFunctor(double slope, double intercept) 
   : Functor("LinearFunctor"), _slope(slope), _intercept(intercept) {
}

PTR(Functor) LinearFunctor::clone() const {
   return boost::make_shared<LinearFunctor>(_slope, _intercept);
}

double LinearFunctor::operator()(double x) const {
   return _slope*x + _intercept;
}

double LinearFunctor::derivative(double x) const {
   return _slope;
}

} // namespace geom
} // namespace afw
} // namespace lsst
