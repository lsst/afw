#ifndef LSST_AFW_MATH_ELLIPSES_AXES_H
#define LSST_AFW_MATH_ELLIPSES_AXES_H

#include <lsst/afw/math/ellipses/Core.h>
#include <lsst/afw/math/ellipses/Ellipse.h>

namespace lsst {
namespace afw {
namespace math {
namespace ellipses {

class Axes : public Core {
public:
    typedef boost::shared_ptr<Axes> Ptr;
    typedef boost::shared_ptr<const Axes> ConstPtr;

    enum Parameters { A=0, B=1, THETA=2 };

    class Ellipse : public lsst::afw::math::ellipses::Ellipse {
        typedef lsst::afw::math::ellipses::Ellipse Super;
    public:
        typedef boost::shared_ptr<Ellipse> Ptr;
        typedef boost::shared_ptr<const Ellipse> ConstPtr;

        enum Parameters { 
            X=Super::X, Y=Super::Y, 
            A=Axes::A+2, B=Axes::B+2, THETA=Axes::THETA+2 
        };

        Axes const & getCore() const {return static_cast<Axes const &>(*_core);}
        Axes & getCore() { return static_cast<Axes &>(*_core); }

        Ellipse * clone() const { return new Ellipse(*this); }

        Ellipse & operator=(Super const & other) {
            return static_cast<Ellipse &>(Super::operator=(other));
        }

        explicit Ellipse(
                Coordinate const & center = Coordinate(0,0)
        ) : Super(center) {}

        template <typename Derived>
        explicit Ellipse(
                Eigen::MatrixBase<Derived> const & vector
        ) : Super(vector.segment<2>(0)) {
            _core.reset(new Axes(vector.segment<3>(2))); 
        }

        explicit Ellipse(
                Axes const & core, 
                Coordinate const & center = Coordinate(0,0)
        ) : Super(core,center) {}

        Ellipse(Super const & other)
            : Super(new Axes(other.getCore()), other.getCenter()) {}

        Ellipse(Ellipse const & other)
            : Super(new Axes(other.getCore()), other.getCenter()) {}
    };


    virtual std::string const getName() const { return "Axes"; }

    virtual void scale(double ratio) { 
        _vector[A] *= ratio; 
        _vector[B] *= ratio; 
    }

    virtual Eigen::Matrix3d getScalingDerivative(double ratio) const { 
        Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
        r(0,0) = r(1,1) = ratio;
        return r;
    }
    
    // swap a,b and rotate if a<b, ensure theta in [-pi/2,pi/2)
    virtual bool normalize(); 

    virtual AffineTransform getGenerator() const;

    virtual Axes * clone() const { return new Axes(*this); }

    virtual Ellipse * makeEllipse(
        Coordinate const & center = Coordinate(0,0)
    ) const;

    template <typename Derived>
    explicit Axes(Eigen::MatrixBase<Derived> const & data) 
        : Core(data) { normalize(); }

    explicit Axes(double a=0, double b=0, double theta=0) 
        : Core(a,b,theta) { normalize(); }

    Axes(Core const & other) { *this = other; }

    Axes & operator=(Axes const & other) { 
        _vector = other._vector; return *this; 
    }
    virtual Axes & operator=(Core const & other) { 
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

#endif // !LSST_AFW_MATH_ELLIPSES_AXES_H
