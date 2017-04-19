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

#ifndef LSST_AFW_GEOM_ELLIPSES_Transformer_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Transformer_h_INCLUDED

/*
 *  Definitions for Ellipse::Transformer and BaseCore::Transformer.
 *
 *  Note: do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/Ellipse.h"
#include "lsst/afw/geom/AffineTransform.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/**
 *  A temporary-only expression object for ellipse core transformations.
 *
 *  Transformer simply provides a clean syntax for transform-related operations, including
 *  in-place and new-object transformations, derivatives of the transformations,
 *  and implicit conversion to a shared_ptr to a new transformed core.
 */
class BaseCore::Transformer {
public:
    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix3d DerivativeMatrix;

    /// Matrix type for derivative with respect to transform parameters.
    typedef Eigen::Matrix<double, 3, 4> TransformDerivativeMatrix;

    /// Standard constructor.
    Transformer(BaseCore &input_, LinearTransform const &transform_) : input(input_), transform(transform_) {}

    /// Return a new transformed ellipse core.
    std::shared_ptr<BaseCore> copy() const;

    /// %Transform the ellipse core in-place.
    void inPlace();

    void apply(BaseCore &result) const;

    /// Return the derivative of transformed core with respect to input core.
    DerivativeMatrix d() const;

    /// Return the derivative of transformed core with respect to transform parameters.
    TransformDerivativeMatrix dTransform() const;

    BaseCore &input;                   ///< input core to be transformed
    LinearTransform const &transform;  ///< transform object
};

/**
 *  A temporary-only expression object for ellipse transformations.
 *
 *  Transformer simply provides a clean syntax for transform-related operations, including
 *  in-place and new-object transformations, derivatives of the transformations, and implicit
 *  conversion to an auto_ptr to a new transformed ellipse.
 */
class Ellipse::Transformer {
public:
    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix<double, 5, 5> DerivativeMatrix;

    /// Matrix type for derivative with respect to transform parameters.
    typedef Eigen::Matrix<double, 5, 6> TransformDerivativeMatrix;

    /// Standard constructor.
    Transformer(Ellipse &input_, AffineTransform const &transform_) : input(input_), transform(transform_) {}

    /// Return a new transformed ellipse.
    std::shared_ptr<Ellipse> copy() const;

    /// %Transform the ellipse in-place.
    void inPlace();

    void apply(Ellipse &other) const;

    /// Return the derivative of transform output ellipse with respect to input ellipse.
    DerivativeMatrix d() const;

    /// Return the derivative of transform output ellipse with respect to transform parameters.
    TransformDerivativeMatrix dTransform() const;

    Ellipse &input;                    ///< input ellipse to be transformed
    AffineTransform const &transform;  ///< transform object
};

inline BaseCore::Transformer BaseCore::transform(LinearTransform const &transform) {
    return BaseCore::Transformer(*this, transform);
}

inline BaseCore::Transformer const BaseCore::transform(LinearTransform const &transform) const {
    return BaseCore::Transformer(const_cast<BaseCore &>(*this), transform);
}

inline Ellipse::Transformer Ellipse::transform(AffineTransform const &transform) {
    return Ellipse::Transformer(*this, transform);
}

inline Ellipse::Transformer const Ellipse::transform(AffineTransform const &transform) const {
    return Ellipse::Transformer(const_cast<Ellipse &>(*this), transform);
}

inline Ellipse::Ellipse(Ellipse::Transformer const &other)
        : _core(other.input.getCore().transform(other.transform.getLinear()).copy()),
          _center(other.transform(other.input.getCenter())) {}
}
}
}
}  // namespace lsst::afw::geom::ellipses

#endif  // !LSST_AFW_GEOM_ELLIPSES_Transformer_h_INCLUDED
