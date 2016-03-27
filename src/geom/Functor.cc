// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include <cmath>
#include <iostream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/Functor.h"

namespace lsst {
namespace afw {
namespace geom {

Functor::Functor(std::string const & name) 
   : daf::base::Citizen(typeid(this)), _name(name) {
}

double Functor::inverse(double y, double tol, unsigned int maxiter) const {
   /// Sanity checks for tol and maxiter.
   if (tol > 1 || tol <= 0) {
      throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                        "tol out-of-range, tol <=0 or tol > 1");
   }
   if (maxiter < 1) {
      throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                        "maxiter out-of-range, maxiter < 1");
   }
   /// Use Newton-Raphson method to find the inverse.
   double x = y;
   for (unsigned int iter=0; iter < maxiter; iter++) {
      double dx = y - operator()(x);
      if (std::fabs(dx) < tol) {
         return x;
      }
      x += dx/derivative(x);
   }
   throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                     "max iteration count exceeded for subclass " + _name);
   return 0;
}

} // namespace geom
} // namespace afw
} // namespace lsst
