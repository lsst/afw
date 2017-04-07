// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2015 LSST Corporation.
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

/*
 * Functor classes used by SeparableXYTransform.
 */

#ifndef LSST_AFW_GEOM_FUNCTOR_H
#define LSST_AFW_GEOM_FUNCTOR_H

#include <string>
#include <memory>
#include "lsst/base.h"
#include "lsst/daf/base.h"

namespace lsst {
namespace afw {
namespace geom {

/** Abstract base class for function objects.
 *
 * A default implementation of the inverse(...) member function is provided
 * that computes the inverse of the function using the Newton-Raphson
 * method.  Concrete subclasses must therefore implement a
 * derivative(...) member function.  In cases where the function is
 * analytically invertible, the inverse(...) function should be
 * re-implemented in the subclass using the analytic expression.
 */
class Functor : public daf::base::Citizen {

public:

   /** @param name The name of the concrete subclass of this class.
    *              Used in the exception message from the inverse(...)
    *              member function.
    */
   explicit Functor(std::string const & name);

   virtual ~Functor() {}

   virtual PTR(Functor) clone() const = 0;

   /// @returns y = f(x)
   virtual double operator()(double x) const = 0;

   /** @param y  desired value of functor
    *  @param tol Convergence tolerance for the Newton-Raphson search
    *             such that abs(x_{iter} - x_{iter-1}) < tol.
    *  @param maxiter Maximum number of iterations in the N-R search.
    *  @returns The x value such that y = f(x).
    *
    *  @throws lsst::pex::exceptions::OutOfRangeError if (tol <= 0) or
    *         (tol > 1) or (maxiter < 1).
    *  @throws lsst::pex::exceptions::RuntimeError if the number of
    *         N-R iterations > maxiter.
    */
   virtual double inverse(double y, double tol=1e-10,
                          unsigned int maxiter=1000) const;

   /** @returns df(x)/dx evaluated at x.  This is used in the
    *          inverse(...) member function.
    */
   virtual double derivative(double x) const = 0;

private:

   std::string _name;

};

/** Concrete implementation of Functor subclass for testing.
 *  Implements a line: y = slope*x + intercept.
 */
class LinearFunctor : public Functor {

public:

   LinearFunctor(double slope, double intercept);

   ~LinearFunctor() {}

   virtual PTR(Functor) clone() const;

   virtual double operator()(double x) const;

   virtual double derivative(double x) const;

private:

   double _slope;
   double _intercept;

};

} // namespace geom
} // namespace af
} // namespace lsst

#endif // LSST_AFW_GEOM_FUNCTOR_H
