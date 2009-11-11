#ifndef LSST_AFW_MATH_ELLIPSES_DISTORTION_H
#define LSST_AFW_MATH_ELLIPSES_DISTORTION_H

#include <lsst/afw/math/ellipses/Core.h>
#include <lsst/afw/math/ellipses/Ellipse.h>

namespace lsst { 
namespace afw {
namespace math {
namespace ellipses {

class Distortion;

class DistortionEllipse : public Ellipse {
public:
    typedef boost::shared_ptr<DistortionEllipse> Ptr;
    typedef boost::shared_ptr<const DistortionEllipse> ConstPtr;

    enum Parameters {X= 0, Y, E1, E2, R};

    Distortion const & getCore() const;
    Distortion & getCore();

    DistortionEllipse * clone() const { return new DistortionEllipse(*this); }

    DistortionEllipse & operator=(Ellipse const & other) {
        return static_cast<DistortionEllipse &>(Ellipse::operator=(other));
    }

    void setComplex(std::complex<double> const & e);
    std::complex<double> getComplex() const;

    void setE(double e);
    double getE() const;
    
    explicit DistortionEllipse(
        lsst::afw::image::PointD const & center = lsst::afw::image::PointD(0,0)
    );
    

    template <typename Derived>
    explicit DistortionEllipse(Eigen::MatrixBase<Derived> const & vector);

    explicit DistortionEllipse(
            Distortion const & core, 
            lsst::afw::image::PointD const & center = 
                lsst::afw::image::PointD(0,0)
    );

    DistortionEllipse(Ellipse const & other);

    DistortionEllipse(DistortionEllipse const & other);
};


class Distortion : public Core {
public:
    typedef boost::shared_ptr<Distortion> Ptr;
    typedef boost::shared_ptr<const Distortion> ConstPtr;
    typedef DistortionEllipse Ellipse;

    enum Parameters { E1=0, E2, R};

    virtual std::string const getName() const { return "Distortion"; }

    virtual void scale(double ratio) { _vector[R] *= ratio; }

    virtual Eigen::Matrix3d differentialScale(double ratio) {
        Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
        r(2,2) = ratio;
        return r;
    }

    virtual Eigen::Matrix3d getScalingDerivative(double ratio) const { 
        Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
        r(2,2) = ratio;
        return r;
    }

    virtual bool normalize() { 
        double e = getE(); return e >= 0.0 && e < 1.0 && _vector[R] >= 0; 
    }

    virtual Distortion * clone() const { return new Distortion(*this); }

    virtual Ellipse * makeEllipse(
        lsst::afw::image::PointD const & center = lsst::afw::image::PointD(0,0)
    ) const;

    void setComplex(std::complex<double> const & e) {
        _vector[E1] = e.real(); _vector[E2] = e.imag(); 
    }
    std::complex<double> getComplex() const { 
        return std::complex<double>(_vector[E1],_vector[E2]); 
    }

    void setE(double e) { 
        double f = e/getE(); _vector[E1] *= f; _vector[E2] *= f; 
    }
    double getE() const { 
        return std::sqrt(_vector[E1]*_vector[E1] + _vector[E2]*_vector[E2]); 
    }

    template <typename Derived>
    explicit Distortion(Eigen::MatrixBase<Derived> const & vector) 
        : Core(vector) {}

    explicit Distortion(double e1=0, double e2=0, double radius=0) 
        : Core(e1,e2,radius) {}
    
    explicit Distortion(std::complex<double> const & e, double radius=0) 
        : Core(e.real(),e.imag(),radius) {}

    Distortion(Core const & other) { *this = other; }

    Distortion & operator=(Distortion const & other) { 
        _vector = other._vector; return *this; 
    }
    virtual Distortion & operator=(Core const & other) { 
        other.assignTo(*this); return *this;     
    }

    virtual Eigen::Matrix3d differentialAssign(Core const & other) {
        return other.differentialAssignTo(*this); 
    }

protected:
    
    virtual void assignTo(Quadrupole & other) const;
    virtual void assignTo(Axes & other) const;
    virtual void assignTo(Distortion & other) const;
    virtual void assignTo(LogShear & other) const;

    virtual Eigen::Matrix3d differentialAssignTo(Quadrupole & other) const;
    virtual Eigen::Matrix3d differentialAssignTo(Axes & other) const;
    virtual Eigen::Matrix3d differentialAssignTo(Distortion & other) const;
    virtual Eigen::Matrix3d differentialAssignTo(LogShear & other) const;
};



}}}} //end namespace lsst::afw::math::ellipses

#endif // !LSST_AFW_MATH_ELLIPSES_DISTORTION_H
