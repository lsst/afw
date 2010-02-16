#ifndef LSST_AFW_GEOM_LINEAR_TRANSFORM_H 
#define LSST_AFW_GEOM_LINEAR_TRANSFORM_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "lsst/pex/exceptions/Runtime.h"

#include "lsst/afw/geom/Point.h"

namespace lsst {
namespace pex {
namespace exceptions {

#ifndef SWIG
LSST_EXCEPTION_TYPE(SingularTransformException, RuntimeErrorException, lsst::pex::exceptions::SingularTransformException)
#endif

}} //namespace pex::exceptions

namespace afw {
namespace geom {

/**
 *  \brief A 2D linear coordinate transformation.
 *
 *  The transform is represented by a matrix \f$ \mathbf{M} \f$ such that
 *  \f[
 *     \left[\begin{array}{ c }
 *     x_f \\
 *     y_f
 *     \end{array}\right]
 *     =
 *     \mathbf{M}
 *     \left[\begin{array}{ c }
 *     x_i \\
 *     y_i
 *     \end{array}\right]
 *  \f]
 *  where \f$(x_i,y_i)\f$ are the input coordinates and \f$(x_f,y_f)\f$ are 
 *  the output coordinates.
 *
 *  If \f$ x_f(x_i,y_i) \f$ and \f$ y_f(x_i,y_i) \f$ are continuous 
 *  differentiable functions, then
 *  \f[
 *     \mathbf{M} = \left[\begin{array}{ c c }
 *     \displaystyle\frac{\partial x_f}{\partial x_i} &
 *     \displaystyle\frac{\partial x_f}{\partial y_i} \\
 *     \displaystyle\frac{\partial y_f}{\partial x_i} &
 *     \displaystyle\frac{\partial y_f}{\partial y_i}
 *     \end{array}\right]
 *  \f]
 *  evaluated at \f$(x_i,y_i)\f$.
 */
class LinearTransform {
public:
    enum Parameters {XX=0,YX=1,XY=2,YY=3};

    typedef Eigen::Matrix<double,4,1> ParameterVector;
    typedef Eigen::Matrix<double,2,4> TransformDerivativeMatrix;
    typedef Eigen::Matrix<double,4,4> ProductDerivativeMatrix;

    typedef Eigen::Matrix<double,2,2,Eigen::DontAlign> Matrix;

    /** \brief Construct an empty (identity) LinearTransform. */
    LinearTransform() : _matrix(Matrix::Identity()) {}

    /** \brief Construct an LinearTransform from an Eigen::Matrix. */
    explicit LinearTransform(Matrix const & matrix) : _matrix(matrix) {}

    LinearTransform operator*(LinearTransform const & other) const {
        return LinearTransform(getMatrix() * other.getMatrix());
    }


    static LinearTransform makeScaling(double s) { 
        return LinearTransform((Matrix() << s, 0.0, 0.0, s).finished());
    }
    
    static LinearTransform makeScaling(double s, double t) { 
        return LinearTransform((Matrix() << s, 0.0, 0.0, t).finished());
    }

    static LinearTransform makeRotation(double t) { 
        return LinearTransform(Matrix(Eigen::Rotation2D<double>(t)));
    }
    
    LinearTransform & operator=(LinearTransform const & other) {
        _matrix = other._matrix;
        return *this;
    }
    
    ParameterVector const getVector() const;
    void setVector(ParameterVector const & vector);
    
    Matrix const & getMatrix() const { return _matrix; }
    Matrix & getMatrix() { return _matrix; }

    double & operator[](int i) { return _matrix(i % 2, i / 2); }
    double const & operator[](int i) const { 
        return const_cast<Matrix&>(_matrix)(i % 2, i / 2);
    }


    LinearTransform const invert() const;

    /** \brief Whether the transform is a no-op. */
    bool isIdentity() const { return getMatrix().isIdentity(); }

    /**
     *  \brief Transform a PointD object. 
     *
     *  This operation is not differentiable, but is faster than apply().
     */
    PointD operator()(PointD const & p) const { return PointD(getMatrix() * p.asVector()); }

    /**
     *  \brief Transform a ExtentD object. 
     *
     *  This operation is not differentiable, but is faster than apply().
     */
    ExtentD operator()(ExtentD const & p) const { return ExtentD(getMatrix() * p.asVector()); }

    TransformDerivativeMatrix dTransform(PointD const & input) const;

    /// Derivative of (*this)(input) with respect to the transform elements (for Extent);
    TransformDerivativeMatrix dTransform(ExtentD const & input) const {
        return dTransform(PointD(input));
    }

    friend std::ostream & operator<<(std::ostream & os, LinearTransform const & t);

private:
    Matrix _matrix;
};

}}} // namespace lsst::afw::geom

#endif // !LSST_AFW_GEOM_LINEAR_TRANSFORM_H
