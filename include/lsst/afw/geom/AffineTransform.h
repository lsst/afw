// -*- lsst-c++ -*-
#ifndef LSST_AFW_MATH_AFFINE_TRANSFORM_H
#define LSST_AFW_MATH_AFFINE_TRANSFORM_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Geometry>
#include <iostream>
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"


namespace lsst {
namespace afw {
namespace geom {

/**
 *  \brief An affine coordinate transformation consisting of a linear transformation and an offset.
 *
 *  The transform is represented by a matrix \f$ \mathbf{M} \f$ such that
 *  \f[
 *     \left[\begin{array}{ c }
 *     x_f \\
 *     y_f \\
 *     1
 *     \end{array}\right]
 *     =
 *     \mathbf{M}
 *     \left[\begin{array}{ c }
 *     x_i \\
 *     y_i \\
 *     1
 *     \end{array}\right]
 *  \f]
 *  where \f$(x_i,y_i)\f$ are the input coordinates and \f$(x_f,y_f)\f$ are the output coordinates.
 *
 *  If \f$ x_f(x_i,y_i) \f$ and \f$ y_f(x_i,y_i) \f$ are continuous differentiable functions, then
 *  \f[
 *     \mathbf{M} = \left[\begin{array}{ c c c }
 *     \displaystyle\frac{\partial x_f}{\partial x_i} &
 *     \displaystyle\frac{\partial x_f}{\partial y_i} &
 *     x_f \\
 *     \displaystyle\frac{\partial y_f}{\partial x_i} &
 *     \displaystyle\frac{\partial y_f}{\partial y_i} &
 *     y_f \\
 *     \displaystyle 0 & \displaystyle 0 & \displaystyle 1
 *     \end{array}\right]
 *  \f]
 *  evaluated at \f$(x_i,y_i)\f$.
 *
 *  The 2x2 upper left corner of \f$ \mathbf{M} \f$ is the linear part of the transform is simply the
 *  Jacobian of the mapping between \f$(x_i,y_i)\f$ and \f$(x_f,y_f)\f$.
 */
class AffineTransform {
    typedef Eigen::Matrix<double, 2, 1, Eigen::RowMajor> EigenPoint;

public:
    typedef boost::shared_ptr<AffineTransform> Ptr;
    typedef boost::shared_ptr<AffineTransform const> ConstPtr;
    typedef lsst::afw::geom::PointD PointD;
    typedef lsst::afw::geom::ExtentD ExtentD;

    typedef Eigen::Transform2d EigenTransform;
    typedef EigenTransform::MatrixType Matrix;
    typedef Eigen::Matrix<double,6,1> ParameterVector;
    typedef Eigen::Matrix<double,2,6> TransformDerivativeMatrix;

    enum Parameters {XX=0,YX=1,XY=2,YY=3,X=4,Y=5};
    /** \brief Construct an empty (identity) AffineTransform. */
    AffineTransform() : _eigenTransform(Eigen::Matrix3d::Identity()) {}

    /** \brief Copy Constructor */
    AffineTransform(const AffineTransform & other) :_eigenTransform(other.getEigenTransform()) {}

    /** \brief Construct an AffineTransform from an EigenTransform. */
    explicit AffineTransform(EigenTransform const & m) : _eigenTransform(m) {}

    /** \brief Construct an AffineTransform from an Eigen::Matrix3d. */
    explicit AffineTransform(Eigen::Matrix3d const & m) : _eigenTransform(m) {}

    /** \brief Construct an AffineTransform with no offset. */
    explicit AffineTransform(Eigen::Matrix2d const & m) : _eigenTransform(m) {}

    /** \brief Construct an AffineTransform from a 2d matrix and offset. */
    explicit AffineTransform(
        Eigen::Matrix2d const & m, 
        ExtentD const & p
    ) : _eigenTransform(Eigen::Translation2d(p.getX(), p.getY()) * EigenTransform(m)) 
    {}

    /** \brief Construct an AffineTransform from an offset and the identity matrix. */
    AffineTransform(ExtentD const & p) : 
        _eigenTransform(Eigen::Translation2d(p.getX(), p.getY()))
    {}

    AffineTransform invert() const;

    /** \brief Whether the transform is a no-op. */
    bool isIdentity() const { return getMatrix().isIdentity(); }


    /** \brief Transform a Point object. */
    PointD operator()(PointD const &p) const {         
        return PointD(_eigenTransform * p.asVector());
    }

    /** \brief Transform an Extent object. */
    ExtentD operator()(ExtentD const &p) const {         
        return ExtentD(_eigenTransform.linear() * p.asVector());
    }

    EigenTransform const & getEigenTransform() const { return _eigenTransform; }
    Matrix const & getMatrix() const { return _eigenTransform.matrix(); }
    
    ParameterVector getVector() const;

    double & operator[](int i) { return _eigenTransform(i % 2, i / 2); }
    double operator[](int i) const { return _eigenTransform(i % 2, i / 2); }

    AffineTransform operator*(AffineTransform const & other) const {
        return AffineTransform(_eigenTransform * other._eigenTransform);
    }

    AffineTransform const & operator =(ParameterVector const & vector);
    AffineTransform const & operator =(EigenTransform const & matrix);
    AffineTransform const & operator =(AffineTransform const & transform);

    /**
     *  \brief Construct a new AffineTransform that represents a uniform scaling.
     *
     *  \return An AffineTransform with matrix
     *  \f$
     *     \left[\begin{array}{ c c c }
     *     s & 0 & 0 \\
     *     0 & s & 0 \\
     *     0 & 0 & 1 \\
     *     \end{array}\right]
     *  \f$
     */
    static AffineTransform makeScaling(double s) { 
        return AffineTransform(EigenTransform(Eigen::Scaling<double,2>(s)));
    }

    /**
     *  \brief Construct a new AffineTransform that represents a CCW rotation in radians.
     *
     *  \return An AffineTransform with matrix
     *  \f$
     *     \left[\begin{array}{ c c c }
     *     \cos t & -\sin t & 0 \\
     *     \sin t & \cos t & 0  \\
     *     0 & 0 & 1 \\
     *     \end{array}\right]
     *  \f$
     */
    static AffineTransform makeRotation(double t) { 
        return AffineTransform(EigenTransform(Eigen::Rotation2D<double>(t)));
    }

    TransformDerivativeMatrix dTransform(PointD const & input) const;
    TransformDerivativeMatrix dTransform(ExtentD const & input) const;

    friend std::ostream & operator<<(std::ostream & os, AffineTransform const & transform);

private:

    EigenTransform _eigenTransform;
};

}}}

#endif // !LSST_AFW_MATH_AFFINE_TRANSFORM_H
