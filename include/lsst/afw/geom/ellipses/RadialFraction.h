// -*- lsst-c++ -*-
#ifndef LSST_AFW_GEOM_ELLIPSES_RADIALFRACTION_H
#define LSST_AFW_GEOM_ELLIPSES_RADIALFRACTION_H

/**
 *  \file
 *  \brief Definitions for BaseEllipse::RadialFraction and BaseCore::RadialFraction.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/Quadrupole.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/** 
 *  \brief Functor for use in evaluating elliptically symmetric functions.
 *
 *  RadialFraction computes, for a given point p, the ratio r of the
 *  radius of the ellipse (with zero centroid) along the ray from the
 *  origin to p, divided by the distance from the origin to p.
 *  Evaluating a radial profile f(r) with unit radius will produce an
 *  elliptically symmetric function, centered at the origin, with that
 *  radial profile and ellipticity and radius matching the
 *  ellipse core.
 */
class BaseCore::RadialFraction {
    Quadrupole::Matrix _inv_matrix;
    Eigen::Matrix3d _jacobian;
public:
    
    typedef boost::shared_ptr<RadialFraction> Ptr;
    typedef boost::shared_ptr<const RadialFraction> ConstPtr;

    typedef Eigen::RowVector2d DerivativeVector;
    typedef Eigen::RowVector3d CoreDerivativeVector;

    typedef Point2D argument_type;
    typedef double result_type;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /// \brief Standard constructor.
    explicit RadialFraction(BaseCore const & core);

    /// \brief Evaluate the RadialFraction at the given point.
    double operator()(Point2D const & p) const {
        return std::sqrt(p.asVector().dot(_inv_matrix * p.asVector()));
    }

    /// \brief Evaluate the gradient (derivative with respect to p).
    DerivativeVector d(Point2D const & p) const;

    /// \brief Evaluate the derivative with respect to the Core parameters.
    CoreDerivativeVector dCore(Point2D const & p) const;

};

/** 
 *  \brief Functor for use in evaluating elliptically symmetric functions.
 *
 *  RadialFraction computes, for a given point p, the ratio r of the
 *  radius of the ellipse (with zero centroid) along the ray from the
 *  center of the ellipse to p, divided by the distance from the
 *  center of the ellipse to p.  Evaluating a radial profile f(r) with
 *  unit radius will produce an elliptically symmetric function, with
 *  that radial profile and centroid, radius, and ellipticity matching
 *  the ellipse.
 */
class BaseEllipse::RadialFraction {
    BaseCore::RadialFraction _coreRF;
    ExtentD _offset;
public:
    
    typedef boost::shared_ptr<RadialFraction> Ptr;
    typedef boost::shared_ptr<const RadialFraction> ConstPtr;

    typedef Eigen::RowVector2d DerivativeVector;
    typedef Eigen::Matrix<double,1,5> EllipseDerivativeVector;

    typedef Point2D argument_type;
    typedef double result_type;

    /// \brief Standard constructor.
    explicit RadialFraction(BaseEllipse const & ellipse) :
        _coreRF(ellipse.getCore()), _offset(Point2D()-ellipse.getCenter()) {}

    /// \brief Evaluate the RadialFraction at the given point.
    double operator()(Point2D const & p) const { return _coreRF(p + _offset); }

    /// \brief Evaluate the gradient (derivative with respect to p).
    DerivativeVector d(Point2D const & p) const { return _coreRF.d(p + _offset); }

    /// \brief Evaluate the derivative with respect to the Ellipse parameters.
    EllipseDerivativeVector dEllipse(Point2D const & p) const;

};

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_RADIALFRACTION_H
