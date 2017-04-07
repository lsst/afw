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
 * Class representing a 2D transform for which the pixel
 * distortions in the x- and y-directions are separable.
 */

#ifndef LSST_AFW_GEOM_SEPARABLEXYTRANSFORM_H
#define LSST_AFW_GEOM_SEPARABLEXYTRANSFORM_H

#include "lsst/afw/geom/XYTransform.h"

namespace lsst {
namespace afw {
namespace geom {

class Functor;

/** @brief A 2D transform for which the pixel distortions in the in
 *  the x- and y-directions are separable.
 *
 *  The transformations in each direction are implemented as separate
 *  instances of concrete subclasses of the Functor base class.
 */
class SeparableXYTransform : public XYTransform {

public:

   /** @param xfunctor Functor describing the transformation from
    *         nominal pixels to actual pixels in the x-direction.
    *  @param yfunctor Functor describing the transformation from
    *         nominal pixels to actual pixels in the y-direction.
    */
   SeparableXYTransform(Functor const & xfunctor, Functor const & yfunctor);

   virtual ~SeparableXYTransform() {}

   virtual PTR(XYTransform) clone() const;

   /**
    * @param point The Point2D location in sensor coordinates in
    *              units of pixels.  This corresponds to the location on
    *              the sensor in the absence of the pixel distortions.
    * @returns The transformed Point2D in sensor coordinates in units
    *         of pixels.
    */
   virtual Point2D forwardTransform(Point2D const & point) const;

   /**
    * @param point The Point2D location in sensor coordinates.  This
    *              corresponds to the actual location of charge deposition,
    *              i.e., with the pixel distortions applied.
    * @returns The un-transformed Point2D in sensor coordinates.
    */
   virtual Point2D reverseTransform(Point2D const & point) const;

   /// @returns Const reference to the xfunctor.
   Functor const & getXfunctor() const;

   /// @returns Const reference to the yfunctor.
   Functor const & getYfunctor() const;

private:

   PTR(Functor) _xfunctor;
   PTR(Functor) _yfunctor;

};

} // namespace geom
} // namespace af
} // namespace lsst

#endif // LSST_AFW_GEOM_SEPARABLEXYTRANSFORM_H
