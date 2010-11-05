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
 
#ifndef LSST_AFW_GEOM_ELLIPSES_TRANSFORMER_H
#define LSST_AFW_GEOM_ELLIPSES_TRANSFORMER_H

/**
 *  \file
 *  \brief Definitions for BaseEllipse::Transformer and BaseCore::Transformer.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include <boost/tuple/tuple.hpp>

#include "lsst/afw/geom/ellipses/BaseEllipse.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Axes.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/LogShear.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/**
 *  A temporary-only expression object for ellipse core transformations.
 *
 *  Transformer simply provides a clean syntax for transform-related operations, including 
 *  in-place and new-object transformations, derivatives of the transformations,
 *  and implicit conversion to an auto_ptr to a new transformed core.
 */
class BaseCore::Transformer {
public:

    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix3d DerivativeMatrix; 

    /// Matrix type for derivative with respect to transform parameters.
    typedef Eigen::Matrix<double,3,4> TransformDerivativeMatrix;

    /// Standard constructor.
    Transformer(BaseCore & input, LinearTransform const & transform) :
        _input(input), _transform(transform) {}

    /// Return a new transformed ellipse core.
    BaseCore::Ptr copy() const;

    /// %Transform the ellipse core in-place.
    void inPlace() { _input = transformQuadrupole(_input); }

    /// Return the derivative of transformed core with respect to input core.
    DerivativeMatrix d() const;
    
    /// Return the derivative of transformed core with respect to transform parameters.
    TransformDerivativeMatrix dTransform() const;

protected:
    BaseCore & _input; ///< \internal input core to be transformed
    LinearTransform const & _transform; ///< \internal transform object

    /// \internal \brief %Transform a Quadrupole core.
    Quadrupole transformQuadrupole(Quadrupole const & input) const;

    /**
     *  \internal \brief Return useful products in computing derivatives of the transform.
     *
     *  \return A tuple of:
     *  - A Quadrupole corresponding to the input Core.
     *  - A Quadrupole corresponding to the transformed Core.
     *  - The Jacobian for the conversion of the input Core to input Quadrupole.
     *  - The Jacobian for the conversion of the output Quadrupole to output Core.
     */
    boost::tuple<Quadrupole,Quadrupole,Jacobian,Jacobian> computeConversionJacobian() const;

};

/**
 *  A temporary-only expression object for ellipse transformations.
 *
 *  Transformer simply provides a clean syntax for transform-related operations, including 
 *  in-place and new-object transformations, derivatives of the transformations, and implicit
 *  conversion to an auto_ptr to a new transformed ellipse.
 */
class BaseEllipse::Transformer {
public:
    
    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix<double,5,5> DerivativeMatrix;

    /// Matrix type for derivative with respect to transform parameters.
    typedef Eigen::Matrix<double,5,6> TransformDerivativeMatrix;

    /// Standard constructor.
    Transformer(BaseEllipse & input, AffineTransform const & transform) :
        _input(input), _transform(transform) {}

    /// Return a new transformed ellipse.
    BaseEllipse::Ptr copy() const;

    /// %Transform the ellipse in-place.
    void inPlace();
    
    /// Return the derivative of transform output ellipse with respect to input ellipse.
    DerivativeMatrix d() const;
    
    /// Return the derivative of transform output ellipse with respect to transform parameters.
    TransformDerivativeMatrix dTransform() const;

protected:
    BaseEllipse & _input; ///< \internal input ellipse to be transformed
    AffineTransform const & _transform; ///< \internal transform object
};

/// Transform the ellipse core by the LinearTransform.
inline BaseCore::Transformer BaseCore::transform(
    LinearTransform const & transform
) {
    return Transformer(*this, transform);
}

/// Transform the ellipse core by the LinearTransform.
inline BaseCore::Transformer const BaseCore::transform(
    LinearTransform const & transform
) const {
    return Transformer(const_cast<BaseCore &>(*this), transform);
}

/// Transform the ellipse core by the linear part of an AffineTransform.
inline BaseCore::Transformer BaseCore::transform(
    AffineTransform const & transform
) {
    return Transformer(*this, transform.getLinear());
}

/// Transform the ellipse core by the linear part of an AffineTransform.
inline BaseCore::Transformer const BaseCore::transform(
    AffineTransform const & transform
) const {
    return Transformer(const_cast<BaseCore &>(*this),transform.getLinear());
}

/// Transform the ellipse by the AffineTransform.
inline BaseEllipse::Transformer BaseEllipse::transform(
    AffineTransform const & transform
) {
    return Transformer(*this, transform);
}

/// Transform the ellipse by the AffineTransform.
inline BaseEllipse::Transformer const BaseEllipse::transform(
    AffineTransform const & transform
) const {
    return Transformer(const_cast<BaseEllipse &>(*this), transform);
}

}}}} // namespace lsst::afw::geom::ellipses
#endif // !LSST_AFW_GEOM_ELLIPSES_TRANSFORMER_H
