#ifndef LSST_AFW_MATH_AFFINE_TRANSFORM_H
#define LSST_AFW_MATH_AFFINE_TRANSFORM_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Geometry>
#include <lsst/afw/image/Utils.h>

namespace Eigen {
typedef Eigen::Matrix<double,6,1> Vector6d;
}

namespace lsst {
namespace afw {
namespace math {

/** \brief Transform defined as the composition of several other distinct Transforms. */
class AffineTransform {
    typedef Eigen::Matrix<double, 2, 1, Eigen::RowMajor> EigenPoint;

public:
    typedef Eigen::Transform2d TransformMatrix; 
    typedef boost::shared_ptr<AffineTransform> Ptr;
    typedef boost::shared_ptr<const AffineTransform> ConstPtr;
    typedef lsst::afw::image::PointD PointD;

    enum Parameters {XX=0,YX=1,XY=2,YY=3,X=4,Y=5};
    /** \brief Construct an empty (identity) AffineTransform. */
    AffineTransform() : _matrix(Eigen::Matrix3d::Identity()) {}

    /** \brief Copy Constructor */
    AffineTransform(const AffineTransform & other) :_matrix(other.matrix()) {}

    /** \brief Construct an AffineTransform from an TransformType. */
    explicit AffineTransform(TransformMatrix const & m) : _matrix(m) {}

    /** \brief Construct an AffineTransform from an Eigen::Matrix3d. */
    explicit AffineTransform(Eigen::Matrix3d const & m) : _matrix(m) {}

    /** \brief Construct an AffineTransform with no offset. */
    explicit AffineTransform(Eigen::Matrix2d const & m) : _matrix(m) {}

    /** \brief Construct an AffineTransform from a 2d matrix and offset. */
    explicit AffineTransform(
        Eigen::Matrix2d const & m, 
        PointD const & p
    ) : _matrix(Eigen::Translation2d(p.getX(), p.getY())*TransformMatrix(m)) 
    {}


    AffineTransform(PointD const & p) : 
        _matrix(Eigen::Translation2d(p.getX(), p.getY()))
    {}

    /** \brief Return a copy of the transform. */
    AffineTransform * clone() const { return new AffineTransform(*this); }

    AffineTransform * invert() const;

    /** \brief Whether the transform is a no-op. */
    bool isIdentity() const { return _matrix.matrix().isIdentity(); }


    /** \brief Transform a Coordinate object. */
    PointD operator()(PointD const &p) const {         
         EigenPoint tp = _matrix * EigenPoint(p.getX(), p.getY());
         return PointD(tp.x(), tp.y());
    }

    TransformMatrix & matrix() {return _matrix;}
    TransformMatrix const & matrix() const {return _matrix;}
    
    Eigen::Vector6d getVector() const;

    double & operator[](int i) { return _matrix(i % 2, i / 2); }
    double const operator[](int i) const { return _matrix(i % 2, i / 2); }

    AffineTransform operator*(AffineTransform const & other) const {
        return AffineTransform(_matrix * other._matrix);
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

    Eigen::Matrix<double,2,6> d(PointD const & input) const;

private:

    TransformMatrix _matrix;
};

}}}

#endif // !LSST_AFW_MATH_AFFINE_TRANSFORM_H
