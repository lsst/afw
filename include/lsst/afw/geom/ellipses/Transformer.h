// -*- lsst-c++ -*-
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
 *  \brief A temporary-only expression object for ellipse core transformations.
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
    typedef Eigen::Matrix<double,3,6> TransformDerivativeMatrix;

protected:

    BaseCore & _input; ///< \internal input core to be transformed
    AffineTransform const & _transform; ///< \internal transform object

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

public:

    /// \brief Standard constructor.
    Transformer(BaseCore & input, AffineTransform const & transform) :
        _input(input), _transform(transform) {}

    /// \brief Return a new transformed ellipse core.
    BaseCore::Ptr copy() const;

    /// \brief %Transform the ellipse core in-place.
    void inPlace() { _input = transformQuadrupole(_input); }

    /// \brief Return the derivative of transformed core with respect to input core.
    DerivativeMatrix d() const;
    
    /// \brief Return the derivative of transformed core with respect to transform parameters.
    TransformDerivativeMatrix dTransform() const;
};

/**
 *  \brief A temporary-only expression object for ellipse transformations.
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

protected:

    BaseEllipse & _input; ///< \internal input ellipse to be transformed
    AffineTransform const & _transform; ///< \internal transform object

public:

    /// \brief Standard constructor.
    Transformer(BaseEllipse & input, AffineTransform const & transform) :
        _input(input), _transform(transform) {}

    /// \brief Return a new transformed ellipse.
    BaseEllipse::Ptr copy() const;

    /// \brief %Transform the ellipse in-place.
    void inPlace();
    
    /// \brief Return the derivative of transform output ellipse with respect to input ellipse.
    DerivativeMatrix d() const;
    
    /// \brief Return the derivative of transform output ellipse with respect to transform parameters.
    TransformDerivativeMatrix dTransform() const;
};

inline BaseCore::Transformer BaseCore::transform(AffineTransform const & transform) {
    return BaseCore::Transformer(*this,transform);
}

inline BaseCore::Transformer const BaseCore::transform(AffineTransform const & transform) const {
    return BaseCore::Transformer(const_cast<BaseCore &>(*this),transform);
}

inline BaseEllipse::Transformer BaseEllipse::transform(AffineTransform const & transform) {
    return BaseEllipse::Transformer(*this,transform);
}

inline BaseEllipse::Transformer const BaseEllipse::transform(AffineTransform const & transform) const {
    return BaseEllipse::Transformer(const_cast<BaseEllipse &>(*this),transform);
}

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_TRANSFORMER_H
