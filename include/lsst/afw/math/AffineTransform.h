#ifndef LSST_AFW_MATH_AFFINE_TRANSFORM_H
#define LSST_AFW_MATH_AFFINE_TRANSFORM_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Geometry>
#include <lsst/afw/math/Coordinate.h>

namespace Eigen {
typedef Eigen::Matrix<double,6,1> Vector6d;
}

namespace lsst {
namespace afw {
namespace math {

/** \brief Transform defined as the composition of several other distinct Transforms. */
class AffineTransform {
public:
    typedef Eigen::Transform2d TransformMatrix; 
    typedef boost::shared_ptr<AffineTransform> Ptr;
    typedef boost::shared_ptr<const AffineTransform> ConstPtr;

    enum Parameters {XX=0,YX=1,XY=2,YY=3,X=4,Y=5};

    /** \brief Construct an empty (identity) AffineTransform. */
    AffineTransform() : _matrix(Eigen::Matrix3d::Identity()) {}

    /** \brief Construct an AffineTransform from an TransformType. */
    explicit AffineTransform(TransformMatrix const & m) : _matrix(m) {}

    /** \brief Construct an AffineTransform from an Eigen::Matrix3d. */
    explicit AffineTransform(Eigen::Matrix3d const & m) : _matrix(m) {}

    /** \brief Construct an AffineTransform with no offset. */
    explicit AffineTransform(Eigen::Matrix2d const & m) : _matrix(m) {}

    /** \brief Construct an AffineTransform from a 2d matrix and offset. */
    explicit AffineTransform(Eigen::Matrix2d const & m, Coordinate const & p) 
        : _matrix(Eigen::Translation2d(p)*TransformMatrix(m)) {}

    /** \brief Construct an AffineTransform with only an offset. */
    AffineTransform(Coordinate const & p) : _matrix(Eigen::Translation2d(p)) {}

    /** \brief Return a copy of the transform. */
    AffineTransform * clone() const { return new AffineTransform(*this); }

    AffineTransform * invert() const;

    /** \brief Whether the transform is a no-op. */
    bool isIdentity() const { return _matrix.matrix().isIdentity(); }

    /** \brief Transform a Coordinate object. */
    Coordinate operator()(const Coordinate& p) const { return _matrix * p; }

    TransformMatrix & matrix() {return _matrix;}
    TransformMatrix const & matrix() const {return _matrix;}
    
    Eigen::Vector6d getVector() const;

    double & operator[](int i) { return _matrix(i % 2, i / 2); }
    double const operator[](int i) const { return _matrix(i % 2, i / 2); }

    friend AffineTransform operator*(
            AffineTransform const & a, 
            AffineTransform const & b
    ) {
        return AffineTransform(a._matrix * b._matrix);
    }

    AffineTransform const & operator =(Eigen::Vector6d const & vector);
    AffineTransform const & operator =(TransformMatrix const & matrix);
    AffineTransform const & operator =(AffineTransform const & transform);

    static AffineTransform makeScaling(double s) { 
        return AffineTransform(TransformMatrix(Eigen::Scaling<double,2>(s)));
    }

    static AffineTransform makeRotation(double t) { 
        return AffineTransform(TransformMatrix(Eigen::Rotation2D<double>(t)));
    }

    Eigen::Matrix<double,2,6> d(Coordinate const & input) const;

private:
    TransformMatrix _matrix;
};

}}}

#endif // !LSST_AFW_MATH_AFFINE_TRANSFORM_H
