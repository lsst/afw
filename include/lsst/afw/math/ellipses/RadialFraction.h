#ifndef LSST_AFW_MATH_ELLIPSES_RADIALFRACTION_H
#define LSST_AFW_MATH_ELLIPSES_RADIALFRACTION_H

#include <lsst/afw/math/ellipses/Quadrupole.h>
#include <lsst/afw/math/ellipses/Ellipse.h>
#include <Eigen/LU>

namespace lsst {
namespace afw {
namespace math {
namespace ellipses {

/** \brief Functor for use in evaluating elliptically symmetric functions.
 *
 *  RadialFraction computes, for a given point p, the squared ratio z
 *  of the radius of the ellipse (with zero centroid) along the ray from the
 *  origin to p, divided 
 *  by the distance from the origin to p.  Evaluating a radial profile f(r)
 *  with unit radius with r = sqrt(z) will produce an elliptically symmetric
 *  function with that radial profile and centroid, size, and ellipticity
 *  matching the ellipse.
 *
 *  For an ellipse with non-zero center, simply use let p = p - ellipse.getCenter().
 */
class RadialFraction {
    typedef Eigen::Matrix<double, 2, 1, Eigen::RowMajor> EigenPoint;
public:
    
    typedef boost::shared_ptr<RadialFraction> Ptr;
    typedef boost::shared_ptr<const RadialFraction> ConstPtr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    RadialFraction(Core const & core) 
        : _inv_matrix(), _jacobian(Eigen::Matrix3d::Identity()) {
        Quadrupole tmp;
        _jacobian = tmp.differentialAssign(core);
        _inv_matrix = tmp.getMatrix().inverse();
    }
	
    /** \brief Evaluate the RadialFraction z at the given point. */
    double operator()(lsst::afw::image::PointD const & p) const {
        EigenPoint e(p.getX(), p.getY());
        return e.dot(_inv_matrix * e);
    }

    /** \brief Evaluate the gradient of RadialFraction (derivative with 
     * respect to p).
     *
     *  The derivative with respect to the center of the ellipse is the 
     *  negative gradient.
     */
    Eigen::RowVector2d differentiateCoordinate(
        lsst::afw::image::PointD const & p
    ) const {
        EigenPoint e(p.getX(), p.getY());
        return Eigen::RowVector2d(2.0*_inv_matrix*e);
    }

    /** \brief Evaluate the derivative of RadialFraction with respect to the Core parameters.
     *  it was initialized with.
     */
    Eigen::RowVector3d differentiateCore(
        lsst::afw::image::PointD const & p
    ) const {
        Eigen::RowVector3d vec;
        EigenPoint e(p.getX(), p.getY());
        EigenPoint tmp1 = _inv_matrix * e;
        QuadrupoleMatrix tmp2;
        tmp2.part<Eigen::SelfAdjoint>() = (tmp1 * tmp1.adjoint()).lazy();
        vec[0] = -tmp2(0,0);
        vec[1] = -tmp2(1,1);
        vec[2] = -2.0*tmp2(1,0);
        return vec * _jacobian;
    }
private:
    QuadrupoleMatrix _inv_matrix;
    Eigen::Matrix3d _jacobian;
};

}}}} //end namespace lsst::afw::math::ellipses

#endif // !LSST_AFW_MATH_ELLIPSES_RADIALFRACTION_H
