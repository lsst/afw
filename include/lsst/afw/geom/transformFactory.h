// -*- lsst-c++ -*-

#ifndef LSST_AFW_GEOM_TRANSFORMFACTORY_H
#define LSST_AFW_GEOM_TRANSFORMFACTORY_H

/*
 * LSST Data Management System
 * Copyright 2008-2017 LSST Corporation.
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
 * Functions for producing Transforms with commonly desired properties.
 */

#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst {
namespace afw {
namespace geom {

/**
 * Approximate a Transform by its local linearization.
 *
 * @tparam FromEndpoint, ToEndpoint The endpoints of the transform.
 *
 * @param original the Transform to linearize
 * @param inPoint the point at which a linear approximation is desired
 * @returns an AffineTransform whose value and Jacobian at `inPoint` match those
 *          of `original`. It may be invertible; in general, linearizations
 *          are invertible if the Jacobian at `inPoint` is invertible.
 *
 * @throws pex::exceptions::InvalidParameterError Thrown if `original` does not
 *             have a well-defined value and Jacobian at `inPoint`
 * @exceptsafe Not exception safe.
 */
AffineTransform linearizeTransform(TransformPoint2ToPoint2 const &original, Point2D const &inPoint);

/**
 * Wrap an AffineTransform as a Transform.
 *
 * @param affine The AffineTransform to wrap.
 * @returns a Transform that that maps any Point2D `x` to `affine(x)`. It shall
 *          be invertible iff `affine` is invertible.
 *
 * @exceptsafe Provides basic exception safety.
 */
TransformPoint2ToPoint2 makeTransform(AffineTransform const &affine);

/**
 * A purely radial polynomial distortion.
 *
 * The Transform transforms an input @f$x@f$ to
 * @f[ \frac{x}{r} \sum_{i=1}^{N} \mathrm{coeffs[i]} \ r^i @f]
 * where @f$r@f$ is the magnitude of @f$x@f$.
 *
 * @param coeffs radial polynomial coefficients. May be an empty vector to
 *               represent the identity transformation; otherwise must have
 *               `size` > 1, `coeffs[0]` = 0, and `coeffs[1]` &ne; 0.
 * @returns the radial distortion represented by `coeffs`. The Transform shall
 *          have an inverse, which may be approximate.
 *
 * @throws pex::exceptions::InvalidParameterError Thrown if `coeffs` does not
 *         have the required format.
 * @exceptsafe Provides basic exception safety.
 */
TransformPoint2ToPoint2 makeRadialTransform(std::vector<double> const &coeffs);

/**
 * A purely radial polynomial distortion.
 *
 * Similar to makeRadialTransform(std::vector<double> const &), but allows the
 * user to provide an inverse.
 *
 * @param forwardCoeffs radial polynomial coefficients. May be an empty vector
 *                      to represent the identity transformation; otherwise
 *                      must have `size` > 1, `coeffs[0]` = 0, and
 *                      `coeffs[1]` &ne; 0.
 * @param inverseCoeffs coefficients for the inverse transform, as above. Does
 *                      not need to have the same degree as `forwardCoeffs`,
 *                      but either both must be empty or neither must be empty.
 * @returns the radial distortion represented by `coeffs`. The Transform shall
 *          have an inverse, whose accuracy is determined by the relationship
 *          between `forwardCoeffs` and `inverseCoeffs`.
 *
 * @throws pex::exceptions::InvalidParameterError Thrown if `forwardCoeffs` or
 *         `inverseCoeffs` does not have the required format.
 * @exceptsafe Provides basic exception safety.
 */
TransformPoint2ToPoint2 makeRadialTransform(std::vector<double> const &forwardCoeffs,
                                            std::vector<double> const &inverseCoeffs);

/**
 * Make an affine transform
 *
 * In the forward direction the returned transform is a shift followed by rotation and scaling:
 * @f$
 *      out = scale \times \left[\begin{array}{ c c }
 *      \cos(rotation) & -\sin(rotation) \\
 *      \sin(rotation) & \cos(rotation)  \\
 *      \end{array}\right] \times (in + shift)
 * @f$
 */
TransformPoint2ToPoint2 makeAffineTransformPoint2(Extent2D const &offset = Extent2D(0, 0),
                                                  Angle const &rotation = 0 * radians, double scale = 1.0);

/**
 * Trivial Transform x &rarr; x.
 *
 * @returns a Transform mapping any Point2D to itself. The Transform's inverse
 *          shall be itself.
 *
 * @exceptsafe Provides basic exception safety.
 */
TransformPoint2ToPoint2 makeIdentityTransform();

}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_GEOM_TRANSFORMFACTORY_H
