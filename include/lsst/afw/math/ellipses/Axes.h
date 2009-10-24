#ifndef LSST_AFW_MATH_ELLIPSES_AXES_H
#define LSST_AFW_MATH_ELLIPSES_AXES_H

#include <lsst/afw/math/ellipses/Core.h>
#include <lsst/afw/math/ellipses/Ellipse.h>

namespace lsst {
namespace afw {
namespace math {
namespace ellipses {

class Axes;

class AxesEllipse : public Ellipse {
public:
    typedef boost::shared_ptr<AxesEllipse> Ptr;
    typedef boost::shared_ptr<const AxesEllipse> ConstPtr;

    enum Parameters {X=0, Y, A, B, THETA};

    Axes const & getCore() const;
    Axes & getCore();

    AxesEllipse * clone() const { return new AxesEllipse(*this); }

    AxesEllipse & operator=(Ellipse const & other) {
        return static_cast<AxesEllipse &>(Ellipse::operator=(other));
    }

    explicit AxesEllipse(
        lsst::afw::math::Coordinate const & center = Coordinate(0,0)
    );

    template <typename Derived>
    explicit AxesEllipse(Eigen::MatrixBase<Derived> const & vector);

    explicit AxesEllipse(
            Axes const & core, 
            lsst::afw::math::Coordinate const & center = Coordinate(0,0)
    );

    AxesEllipse(Ellipse const & other);

    AxesEllipse(AxesEllipse const & other);
};

class Axes : public Core {
public:
    typedef boost::shared_ptr<Axes> Ptr;
    typedef boost::shared_ptr<const Axes> ConstPtr;
    typedef AxesEllipse Ellipse;

    enum Parameters { A=0, B=1, THETA=2 };

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
    explicit Axes(
            Eigen::MatrixBase<Derived> const & data
    ) : Core(data) { 
        normalize(); 
    }

    Axes() {}

    Axes(
        double a, double b, double theta, bool doNormalize = true
    ) : Core(a,b,theta) { 
        if(doNormalize)
            normalize(); 
    }

    Axes(Core const & other) { *this = other; }

    Axes & operator=(Axes const & other) { 
        _vector = other._vector; 
        return *this; 
    }
    virtual Axes & operator=(Core const & other) { 
        other.assignTo(*this); 
        return *this; 
    }

    virtual Eigen::Matrix3d differentialAssign(Core const & other) {
        return other.differentialAssignTo(*this); 
    }

    ~Axes() {}
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
