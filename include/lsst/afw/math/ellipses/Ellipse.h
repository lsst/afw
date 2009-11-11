#ifndef LSST_AFW_MATH_ELLIPSES_ELLIPSE_H
#define LSST_AFW_MATH_ELLIPSES_ELLIPSE_H

#include <boost/scoped_ptr.hpp>

#include <lsst/afw/math/ellipses/Core.h>

namespace Eigen {
typedef Matrix<double,5,1> Vector5d;
typedef Matrix<double,5,5> Matrix5d;
}

namespace lsst {
namespace afw {
namespace math { 

class Rectangle;

namespace ellipses {

class Ellipse {
public:
#if !defined SWIG
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    class TransformDerivative {
        lsst::afw::image::PointD _center;
        Core::TransformDerivative _core_d;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        typedef boost::shared_ptr<TransformDerivative> Ptr;
        typedef boost::shared_ptr<const TransformDerivative> ConstPtr;

        TransformDerivative(
            Ellipse const & ellipse, 
            AffineTransform const & transform
        );

        // Derivative of transform output ellipse with respect to input ellipse.
        Eigen::Matrix5d dInput() const;

        // Derivative of transform output ellipse with respect to 
        // transform parameters.
        Eigen::Matrix<double,5,6> dTransform() const;
    };


    typedef boost::shared_ptr<Ellipse> Ptr;
    typedef boost::shared_ptr<const Ellipse> ConstPtr;
#endif
    enum Parameters { X=0, Y=1 };

    lsst::afw::image::PointD const & getCenter() const { return _center; }
    void setCenter(lsst::afw::image::PointD const & center) {_center = center; }

    Core const & getCore() const { return *_core; }
    Core & getCore() { return *_core; }

    bool normalize() { return _core->normalize(); }

    virtual Ellipse * clone() const = 0;

    virtual ~Ellipse() {}

    void transform(AffineTransform const & transform);

    lsst::afw::math::AffineTransform getGenerator() const;

    //Rectangle getEnvelope() const;


    double & operator[](int i) { return (i<2) ? _center[i] : (*_core)[i-2]; }
    double const operator[](int i) const { 
        return (i<2) ? _center[i] : (*_core)[i-2]; 
    }

    Ellipse & operator=(Ellipse const & other) { 
        _center = other.getCenter();
        *_core = other.getCore();
        return *this;
    }


#if !defined SWIG
    Eigen::Vector5d const getVector() const {
        Eigen::Vector5d r;
        r << _center[X], _center[Y], (*_core)[0], (*_core)[1], (*_core)[2];
        return r;
    }

    template <typename Derived>
    void setVector(Eigen::MatrixBase<Derived> const & vector) {
        _center = lsst::afw::image::PointD(vector[0], vector[1]);
        _core->setVector(vector.template segment<3>(2));
    }
#endif
protected:    
    explicit Ellipse(lsst::afw::image::PointD const & center) 
        : _center(center), _core() {}

    explicit Ellipse(Core const & core, lsst::afw::image::PointD const & center)
        : _center(center), _core(core.clone()) {}

    explicit Ellipse(Core * core, lsst::afw::image::PointD const & center)
        : _center(center), _core(core) {}

    lsst::afw::image::PointD _center;
    boost::scoped_ptr<Core> _core;
};


/** 
 * @brief Functor that returns points on the boundary of the ellipse as a 
 *   function of a parameter that runs between 0 and 2/pi (but is not angle).
 */
class Parametric {
public:
    typedef boost::shared_ptr<Parametric> Ptr;
    typedef boost::shared_ptr<const Parametric> ConstPtr;

    Parametric(Ellipse const & ellipse);

    lsst::afw::image::PointD operator()(double t) const {
        double c = std::cos(t);
        double s = std::sin(t);
        return lsst::afw::image::PointD(
            _center.getX() + _u.getX()*c + _v.getX()*s,
            _center.getY() + _u.getY()*c + _v.getY()*s    
        );
    }
private:
    lsst::afw::image::PointD _center;
    lsst::afw::image::PointD _u;
    lsst::afw::image::PointD _v;
};

}}}} //end namespace lsst::afw::math::ellipses

#endif // !LSST_AFW_MATH_ELLIPSES_ELLIPSE_H
