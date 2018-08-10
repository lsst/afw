// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

#ifndef LSST_AFW_GEOM_ELLIPSES_GridTransform_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_GridTransform_h_INCLUDED

/*
 *  Definitions for Ellipse::GridTransform and BaseCore::GridTransform.
 *
 *  Note: do not include directly; use the main ellipse header file.
 */

#include "Eigen/Eigenvalues"

#include "lsst/afw/geom/ellipses/Ellipse.h"
#include "lsst/geom/AffineTransform.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/**
 *  @brief A temporary-only expression object representing an lsst::geom::LinearTransform that
 *         maps the ellipse core to a unit circle.
 */
class BaseCore::GridTransform {
public:
    /// Matrix type for derivative with respect to ellipse parameters.
    typedef Eigen::Matrix<double, 4, 3> DerivativeMatrix;

    /// Standard constructor.
    explicit GridTransform(BaseCore const& input);

    /// Convert the proxy to an lsst::geom::LinearTransform.
    operator lsst::geom::LinearTransform() const;

    /// Return the transform matrix as an Eigen object.
    lsst::geom::LinearTransform::Matrix getMatrix() const;

    /// Return the derivative of the transform with respect to input core.
    DerivativeMatrix d() const;

    /// Return the determinant of the lsst::geom::LinearTransform.
    double getDeterminant() const;

    //@{
    /**
     * Return the inverse of the lsst::geom::LinearTransform;
     *
     * @deprecated invert is deprecated in favor of inverted
     */
    lsst::geom::LinearTransform inverted() const;
    lsst::geom::LinearTransform invert() const { return inverted(); };
    //@}

private:
    BaseCore const& _input;  ///< @internal input core to be transformed
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> _eig;
};

/**
 *  @brief A temporary-only expression object representing an lsst::geom::AffineTransform that
 *         maps the Ellipse to a unit circle at the origin.
 */
class Ellipse::GridTransform {
public:
    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix<double, 6, 5> DerivativeMatrix;

    /// Standard constructor.
    explicit GridTransform(Ellipse const& input);

    /// Return the transform matrix as an Eigen object.
    lsst::geom::AffineTransform::Matrix getMatrix() const;

    /// Return the derivative of transform with respect to input ellipse.
    DerivativeMatrix d() const;

    /// Return the determinant of the lsst::geom::AffineTransform.
    double getDeterminant() const;

    /// Convert the proxy to an lsst::geom::AffineTransform.
    operator lsst::geom::AffineTransform() const;

    /// Return the inverse of the AffineTransform.
    lsst::geom::AffineTransform inverted() const;
    lsst::geom::AffineTransform invert() const { return inverted(); };

private:
    Ellipse const& _input;  ///< @internal input ellipse to be transformed
    BaseCore::GridTransform _coreGt;
};

inline BaseCore::GridTransform const BaseCore::getGridTransform() const {
    return BaseCore::GridTransform(*this);
}

inline Ellipse::GridTransform const Ellipse::getGridTransform() const {
    return Ellipse::GridTransform(*this);
}
}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_GEOM_ELLIPSES_GridTransform_h_INCLUDED
