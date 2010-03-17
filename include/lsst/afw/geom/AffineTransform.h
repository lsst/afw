// -*- lsst-c++ -*-
#ifndef LSST_AFW_MATH_AFFINE_TRANSFORM_H
#define LSST_AFW_MATH_AFFINE_TRANSFORM_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Geometry>
#include <iostream>
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/LinearTransform.h"

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
public:
    typedef boost::shared_ptr<AffineTransform> Ptr;
    typedef boost::shared_ptr<AffineTransform const> ConstPtr;

    enum Parameters {XX=0,YX=1,XY=2,YY=3,X=4,Y=5};

    typedef Eigen::Matrix3d Matrix;
    typedef Eigen::Matrix<double,6,1> ParameterVector;
    typedef Eigen::Matrix<double,2,6> TransformDerivativeMatrix;


    /** Construct an empty (identity) AffineTransform. */
    AffineTransform() : _linear(), _translation() {}

    /** Construct an AffineTransform from a 3x3 matrix. */
    explicit AffineTransform(Eigen::Matrix3d const & matrix) 
      : _linear(matrix.block<2,2>(0,0)), 
        _translation(matrix.block<2,1>(0,2)) 
    {}

    /** Construct an AffineTransform with no translation from a 2x2 matrix. */
    explicit AffineTransform(Eigen::Matrix2d const & linear) 
      : _linear(linear), _translation() {}

    /** Construct a translation-only AffineTransform from a vector. */
    explicit AffineTransform(Eigen::Vector2d const & translation) 
      : _linear(), _translation(translation) {}

    /** Construct an AffineTransform from a 2x2 matrix and vector. */
    explicit AffineTransform(
        Eigen::Matrix2d const & linear, Eigen::Vector2d const & translation
    ) : _linear(linear), _translation(translation) {}
    
    /** Construct an AffineTransform from a LinearTransform. */
    explicit AffineTransform(LinearTransform const & linear) 
      : _linear(linear), _translation() {}

    /** Construct a translation-only AffineTransform from an ExtentD. */
    explicit AffineTransform(ExtentD const & translation) 
      : _linear(), _translation(translation) {}

    /** Construct an AffineTransform from a LinearTransform and ExtentD. */
    explicit AffineTransform(
        LinearTransform const & linear, ExtentD const & translation
    ) : _linear(linear), _translation(translation) {}


    AffineTransform const invert() const;

    /** Whether the transform is a no-op. */
    bool isIdentity() const { return getMatrix().isIdentity(); }


    /** 
     * Transform a Point object. 
     *
     * The result is affected by the translation parameters of the transform
     */
    PointD operator()(PointD const &p) const {         
        return PointD(_linear(p) + _translation);
    }

    /**
     * Transform an Extent object. 
     *
     * The result is unaffected by the translation parameters of the transform
     */
    ExtentD operator()(ExtentD const &p) const {         
        return ExtentD(_linear(p));
    }

    ExtentD const & getTranslation() const {return _translation;}
    ExtentD & getTranslation() {return _translation;}

    LinearTransform const & getLinear() const {return _linear;}
    LinearTransform & getLinear() {return _linear;}

    Matrix const getMatrix() const;
    
    ParameterVector const getVector() const;
    void setVector(ParameterVector const & vector);

    double & operator[](int i) { 
        return (i < 4) ? _linear[i] : _translation[i - 4]; 
    }
    double const operator[](int i) const { 
        return (i < 4) ? _linear[i] : _translation[i - 4]; 
    }

    AffineTransform operator*(AffineTransform const & other) const {
        return AffineTransform(
            getLinear()*other.getLinear(),
            getLinear()(other.getTranslation()) + getTranslation()
        );
    }

    AffineTransform & operator =(AffineTransform const & other) {
        _linear = other._linear;
        _translation = other._translation;
        return *this;
    }

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
        return AffineTransform(LinearTransform::makeScaling(s));
    }

    /**
     *  \brief Construct a new AffineTransform that represents a non-uniform 
     *  scaling.
     *
     *  \return An AffineTransform with matrix
     *  \f$
     *     \left[\begin{array}{ c c c }
     *     s & 0 & 0 \\
     *     0 & t & 0 \\
     *     0 & 0 & 1 \\
     *     \end{array}\right]
     *  \f$
     */
    static AffineTransform makeScaling(double s, double t) { 
        return AffineTransform(LinearTransform::makeScaling(s, t));
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
        return AffineTransform(LinearTransform::makeRotation(t));
    }

    /**
     *  \brief Construct a new AffineTransform that represents a pure translation.
     *
     *  \return An AffineTransform with matrix
     *  \f$
     *     \left[\begin{array}{ c c c }
     *     0 & 0 & translation.getX() \\
     *     0 & 0 & translation.getY() \\
     *     0 & 0 & 1 \\
     *     \end{array}\right]
     *  \f$
     */
    static AffineTransform makeTranslation(Extent2D translation) { 
        return AffineTransform(translation);
    }

    TransformDerivativeMatrix dTransform(PointD const & input) const;
    TransformDerivativeMatrix dTransform(ExtentD const & input) const;

    friend std::ostream & operator<<(std::ostream & os, AffineTransform const & transform);

private:

    LinearTransform _linear;
    ExtentD _translation;
};

}}}

#endif // !LSST_AFW_MATH_AFFINE_TRANSFORM_H
