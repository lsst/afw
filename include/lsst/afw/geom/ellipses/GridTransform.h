// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_GEOM_ELLIPSES_GridTransform_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_GridTransform_h_INCLUDED

/**
 *  @file
 *  @brief Definitions for Ellipse::GridTransform and BaseCore::GridTransform.
 *
 *  @note Do not include directly; use the main ellipse header file.
 */

#include "Eigen/Eigenvalues"

#include "lsst/afw/geom/ellipses/Ellipse.h"
#include "lsst/afw/geom/AffineTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief A temporary-only expression object representing a LinearTransform that
 *         maps the ellipse core to a unit circle.
 */
class BaseCore::GridTransform {
public:

    /// Matrix type for derivative with respect to ellipse parameters.
    typedef Eigen::Matrix<double,4,3> DerivativeMatrix;

    /// @brief Standard constructor.
    explicit GridTransform(BaseCore const & input);
    
    /// @brief Convert the proxy to a LinearTransform.
    operator LinearTransform () const;

    /// @brief Return the transform matrix as an Eigen object.
    LinearTransform::Matrix getMatrix() const;

    /// @brief Return the derivative of the transform with respect to input core.
    DerivativeMatrix d() const;

    /// @brief Return the determinant of the LinearTransform.
    double getDeterminant() const;

    /// @brief Return the inverse of the LinearTransform;
    LinearTransform invert() const;

private:

    BaseCore const & _input; ///< \internal input core to be transformed
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> _eig;
};

/**
 *  @brief A temporary-only expression object representing an AffineTransform that
 *         maps the Ellipse to a unit circle at the origin.
 */
class Ellipse::GridTransform {
public:
    
    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix<double,6,5> DerivativeMatrix;

    /// @brief Standard constructor.
    explicit GridTransform(Ellipse const & input);

    /// @brief Return the transform matrix as an Eigen object.
    AffineTransform::Matrix getMatrix() const;
    
    /// @brief Return the derivative of transform with respect to input ellipse.
    DerivativeMatrix d() const;
    
    /// @brief Return the determinant of the AffineTransform.
    double getDeterminant() const;

    /// @brief Convert the proxy to a AffineTransform.
    operator AffineTransform () const;

    /// @brief Return the inverse of the AffineTransform.
    AffineTransform invert() const;

private:

    Ellipse const & _input; ///< \internal input ellipse to be transformed
    BaseCore::GridTransform _coreGt;
};

inline BaseCore::GridTransform const BaseCore::getGridTransform() const{
    return BaseCore::GridTransform(*this);
}

inline Ellipse::GridTransform const Ellipse::getGridTransform() const {
    return Ellipse::GridTransform(*this);
}

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_GridTransform_h_INCLUDED
