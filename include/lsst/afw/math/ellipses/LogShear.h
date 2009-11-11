#ifndef LSST_AFW_MATH_ELLIPSES_LOGSHEAR_H
#define LSST_AFW_MATH_ELLIPSES_LOGSHEAR_H

#include <lsst/afw/math/ellipses/Core.h>
#include <lsst/afw/math/ellipses/Ellipse.h>

namespace lsst {
namespace afw {
namespace math{ 
namespace ellipses {

class LogShear;

class LogShearEllipse : public Ellipse {
public:
    typedef boost::shared_ptr<LogShearEllipse> Ptr;
    typedef boost::shared_ptr<const LogShearEllipse> ConstPtr;

    enum Parameters {X=0, Y,GAMMA1, GAMMA2, KAPPA};

    LogShear const & getCore() const;
    LogShear & getCore();
    
    LogShearEllipse * clone() const { return new LogShearEllipse(*this); }

    void setComplex(std::complex<double> const & gamma);
    std::complex<double> getComplex() const;
    void setGamma(double gamma);
    double getGamma() const;

    LogShearEllipse & operator=(Ellipse const & other) {
        return static_cast<LogShearEllipse &>(Ellipse::operator=(other));
    }

    explicit LogShearEllipse(
        lsst::afw::image::PointD const & center = lsst::afw::image::PointD(0,0)
    );

    template <typename Derived>
    explicit LogShearEllipse(Eigen::MatrixBase<Derived> const & vector);

    explicit LogShearEllipse(
        LogShear const & core, 
        lsst::afw::image::PointD const & center = lsst::afw::image::PointD(0,0)
    );

    LogShearEllipse(Ellipse const & other);

    LogShearEllipse(LogShearEllipse const & other);
};


class LogShear : public Core {
public:
    typedef boost::shared_ptr<LogShear> Ptr;
    typedef boost::shared_ptr<const LogShear> ConstPtr;
    typedef LogShearEllipse Ellipse;

    enum Parameters { GAMMA1=0, GAMMA2=1, KAPPA=2 };
    
    virtual std::string const getName() const { return "LogShear"; }

    virtual void scale(double ratio) { _vector[KAPPA] += std::log(ratio); }

    virtual Eigen::Matrix3d getScalingDerivative(double ratio) const {
        return Eigen::Matrix3d::Identity();
    }

    virtual bool normalize() { return true; }

    virtual LogShear * clone() const { return new LogShear(*this); }

    virtual Ellipse * makeEllipse(
        lsst::afw::image::PointD const & center = lsst::afw::image::PointD(0,0)
    ) const;

    void setComplex(std::complex<double> const & gamma) { 
        _vector[GAMMA1] = gamma.real(); _vector[GAMMA2] = gamma.imag(); 
    }
    std::complex<double> getComplex() const { 
        return std::complex<double>(_vector[GAMMA1],_vector[GAMMA2]); 
    }

    void setGamma(double gamma) { 
        double f=gamma/getGamma(); _vector[GAMMA1] *= f; _vector[GAMMA2] *= f; 
    }
    double getGamma() const {
        return std::sqrt(
            _vector[GAMMA1]*_vector[GAMMA1] + _vector[GAMMA2]*_vector[GAMMA2]
        ); 
    }

    template <typename Derived>
    explicit LogShear(Eigen::MatrixBase<Derived> const & vector) 
        : Core(vector) {}

    explicit LogShear(
        double gamma1=0, 
        double gamma2=0,
        double kappa=-std::numeric_limits<double>::infinity()
    ) : Core(gamma1,gamma2,kappa) {}

    explicit LogShear(
        std::complex<double> const & gamma,
        double kappa=-std::numeric_limits<double>::infinity()
    ) : Core(gamma.real(),gamma.imag(),kappa) {}

    LogShear(Core const & other) { *this = other; }

    LogShear & operator=(LogShear const & other) { 
        _vector = other._vector; return *this; 
    }
    virtual LogShear & operator=(Core const & other) {
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

#endif // !LSST_AFW_MATH_ELLIPSES_LOGSHEAR_H
