#ifndef LSST_AFW_MATH_AFFINE_TRANSFORM_H
#define LSST_AFW_MATH_AFFINE_TRANSFORM_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Geometry>
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace lsst {
namespace afw {
namespace geom {

/** \brief Transform defined as the composition of several other distinct Transforms. */
class AffineTransform {
    typedef Eigen::Matrix<double, 2, 1, Eigen::RowMajor> EigenPoint;

public:
    typedef Eigen::Transform2d TransformMatrix; 
    typedef boost::shared_ptr<AffineTransform> Ptr;
    typedef boost::shared_ptr<AffineTransform const> ConstPtr;
    typedef lsst::afw::geom::PointD PointD;
    typedef lsst::afw::geom::ExtentD ExtentD;

    typedef Eigen::Matrix<double,6,1> ParameterVector;
    typedef Eigen::Matrix<double,2,6> TransformDerivativeMatrix;

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
        ExtentD const & p
    ) : _matrix(Eigen::Translation2d(p.getX(), p.getY())*TransformMatrix(m)) 
    {}


    AffineTransform(ExtentD const & p) : 
        _matrix(Eigen::Translation2d(p.getX(), p.getY()))
    {}

    AffineTransform invert() const;

    /** \brief Whether the transform is a no-op. */
    bool isIdentity() const { return _matrix.matrix().isIdentity(); }


    /** \brief Transform a Point object. */
    PointD operator()(PointD const &p) const {         
        EigenPoint tp = _matrix * EigenPoint(p.getX(), p.getY());
        return PointD(tp.x(), tp.y());
    }

    /** \brief Transform an Extent object. */
    ExtentD operator()(ExtentD const &p) const {         
        EigenPoint tp = _matrix.linear() * EigenPoint(p.getX(), p.getY());
        return ExtentD(tp.x(), tp.y());
    }

    TransformMatrix & matrix() {return _matrix;}
    TransformMatrix const & matrix() const {return _matrix;}
    
    ParameterVector getVector() const;

    double & operator[](int i) { return _matrix(i % 2, i / 2); }
    double const operator[](int i) const { return _matrix(i % 2, i / 2); }

    AffineTransform operator*(AffineTransform const & other) const {
        return AffineTransform(_matrix * other._matrix);
    }


    AffineTransform const & operator =(ParameterVector const & vector);
    AffineTransform const & operator =(TransformMatrix const & matrix);
    AffineTransform const & operator =(AffineTransform const & transform);

    static AffineTransform makeScaling(double s) { 
        return AffineTransform(TransformMatrix(Eigen::Scaling<double,2>(s)));
    }

    static AffineTransform makeRotation(double t) { 
        return AffineTransform(TransformMatrix(Eigen::Rotation2D<double>(t)));
    }

    TransformDerivativeMatrix dTransform(PointD const & input) const;
    TransformDerivativeMatrix dTransform(ExtentD const & input) const;

private:

    TransformMatrix _matrix;
};

}}}

#endif // !LSST_AFW_MATH_AFFINE_TRANSFORM_H
