#ifndef LSST_AFW_MATH_ELLIPSES_DISTORTION_H
#define LSST_AFW_MATH_ELLIPSES_DISTORTION_H

#include <lsst/afw/math/ellipses/Core.h>
#include <lsst/afw/math/ellipses/Ellipse.h>

namespace lsst { 
namespace afw {
namespace math {
namespace ellipses {

class Distortion : public Core {
public:
    typedef boost::shared_ptr<Distortion> Ptr;
    typedef boost::shared_ptr<const Distortion> ConstPtr;

    enum Parameters { E1=0, E2=1, R=2 };

    class Ellipse : public lsst::afw::math::ellipses::Ellipse {
        typedef lsst::afw::math::ellipses::Ellipse Super;
    public:

        typedef boost::shared_ptr<Ellipse> Ptr;
        typedef boost::shared_ptr<const Ellipse> ConstPtr;

        enum Parameters {
            X=Super::X, Y=Super::Y, 
            E1=Distortion::E1+2, E2=Distortion::E2+2, R=Distortion::R+2 
        };

        Distortion const & getCore() const { return static_cast<Distortion const &>(*_core); }
        Distortion & getCore() { return static_cast<Distortion &>(*_core); }

        Ellipse * clone() const { return new Ellipse(*this); }

        Ellipse & operator=(Super const & other) {
            return static_cast<Ellipse &>(Super::operator=(other));
        }

        void setComplex(std::complex<double> const & e) { getCore().setComplex(e); }
        std::complex<double> getComplex() const { return getCore().getComplex(); }

        void setE(double e) { getCore().setE(e); }
        double getE() const { return getCore().getE(); }
    
        explicit Ellipse(
                Coordinate const & center = Coordinate(0,0)
        ) : Super(center) {}

        template <typename Derived>
        explicit Ellipse(
            Eigen::MatrixBase<Derived> const & vector
        ) : Super(vector.segment<2>(0)) {
            _core.reset(new Distortion(vector.segment<3>(2)));
        }

        explicit Ellipse(
                Distortion const & core, 
                Coordinate const & center = Coordinate(0,0)
        ) : Super(core,center) {}

        Ellipse(Super const & other)
            : Super(new Distortion(other.getCore()), other.getCenter()) {}

        Ellipse(Ellipse const & other)
            : Super(new Distortion(other.getCore()), other.getCenter()) {}
    };
    
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

    virtual bool normalize() { double e = getE(); return e >= 0.0 && e < 1.0 && _vector[R] >= 0; }

    virtual Distortion * clone() const { return new Distortion(*this); }

    virtual Ellipse * makeEllipse(
            Coordinate const & center = Coordinate(0,0)
    ) const;

    void setComplex(std::complex<double> const & e) {
        _vector[E1] = e.real(); _vector[E2] = e.imag(); 
    }
    std::complex<double> getComplex() const { 
        return std::complex<double>(_vector[E1],_vector[E2]); 
    }

    void setE(double e) { double f = e/getE(); _vector[E1] *= f; _vector[E2] *= f; }
    double getE() const { return std::sqrt(_vector[E1]*_vector[E1] + _vector[E2]*_vector[E2]); }

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
