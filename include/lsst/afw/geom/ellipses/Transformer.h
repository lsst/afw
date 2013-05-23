// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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

#ifndef LSST_AFW_GEOM_ELLIPSES_Transformer_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Transformer_h_INCLUDED

#include "lsst/afw/geom/ellipses/Ellipse.h"
#include "lsst/afw/geom/AffineTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief A temporary-only expression object for ellipse core transformations.
 *
 *  Transformer simply provides a clean syntax for transform-related operations, including
 *  in-place and new-object transformations, derivatives of the transformations,
 *  and implicit conversion to a shared_ptr to a new transformed core.
 */
class EllipseCore::Transformer {
public:

    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix3d DerivativeMatrix;

    /// Matrix type for derivative with respect to transform parameters.
    typedef Eigen::Matrix<double,3,4> TransformDerivativeMatrix;

    /// Standard constructor.
    Transformer(EllipseCore & input_, LinearTransform const & transform_) :
        input(input_), transform(transform_) {}

    /// Return a new transformed ellipse core.
    PTR(EllipseCore) copy() const;

    /// %Transform the ellipse core in-place.
    void inPlace();

    void apply(EllipseCore & result) const;

    /// Return the derivative of transformed core with respect to input core.
    DerivativeMatrix d() const;

    /// Return the derivative of transformed core with respect to transform parameters.
    TransformDerivativeMatrix dTransform() const;

    EllipseCore & input; ///< input core to be transformed
    LinearTransform const & transform; ///< transform object

};

/**
 *  @brief A temporary-only expression object for ellipse transformations.
 *
 *  Transformer simply provides a clean syntax for transform-related operations, including
 *  in-place and new-object transformations, derivatives of the transformations, and implicit
 *  conversion to an auto_ptr to a new transformed ellipse.
 */
class Ellipse::Transformer {
public:

    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix<double,5,5> DerivativeMatrix;

    /// Matrix type for derivative with respect to transform parameters.
    typedef Eigen::Matrix<double,5,6> TransformDerivativeMatrix;

    /// Standard constructor.
    Transformer(Ellipse & input_, AffineTransform const & transform_) :
        input(input_), transform(transform_) {}

    /// Return a new transformed ellipse.
    Ellipse copy() const;

    /// %Transform the ellipse in-place.
    void inPlace();

    void apply(Ellipse & other) const;

    /// Return the derivative of transform output ellipse with respect to input ellipse.
    DerivativeMatrix d() const;

    /// Return the derivative of transform output ellipse with respect to transform parameters.
    TransformDerivativeMatrix dTransform() const;

    Ellipse & input; ///< input ellipse to be transformed
    AffineTransform const & transform; ///< transform object
};

inline EllipseCore::Transformer EllipseCore::transform(LinearTransform const & transform) {
    return EllipseCore::Transformer(*this,transform);
}

inline EllipseCore::Transformer const EllipseCore::transform(LinearTransform const & transform) const {
    return EllipseCore::Transformer(const_cast<EllipseCore &>(*this),transform);
}

inline Ellipse::Transformer Ellipse::transform(AffineTransform const & transform) {
    return Ellipse::Transformer(*this,transform);
}

inline Ellipse::Transformer const Ellipse::transform(AffineTransform const & transform) const {
    return Ellipse::Transformer(const_cast<Ellipse &>(*this),transform);
}

inline Ellipse::Ellipse(Ellipse::Transformer const & other) :
    _core(other.input.getCore().transform(other.transform.getLinear()).copy()),
    _center(other.transform(other.input.getCenter()))
{}

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Transformer_h_INCLUDED
