#ifndef LSST_AFW_MATH_ELLIPSES_LOGSHEAR_H
#define LSST_AFW_MATH_ELLIPSES_LOGSHEAR_H

#include <lsst/afw/math/ellipses/Core.h>
#include <lsst/afw/math/ellipses/Ellipse.h>

namespace lsst {
namespace afw {
namespace math{ 
namespace ellipses {


class LogShear : public Core {
public:
    typedef boost::shared_ptr<LogShear> Ptr;
    typedef boost::shared_ptr<const LogShear> ConstPtr;

    enum Parameters { GAMMA1=0, GAMMA2=1, KAPPA=2 };

    class Ellipse : public lsst::afw::math::ellipses::Ellipse {
        typedef lsst::afw::math::ellipses::Ellipse Super;
    public:
        typedef boost::shared_ptr<Ellipse> Ptr;
        typedef boost::shared_ptr<const Ellipse> ConstPtr;

        enum Parameters { 
            X=Super::X, Y=Super::Y, 
            GAMMA1=LogShear::GAMMA1+2, 
            GAMMA2=LogShear::GAMMA2+2, 
            KAPPA=LogShear::KAPPA+2
        };

        LogShear const & getCore() const { 
            return static_cast<LogShear const &>(*_core); 
        }
        LogShear & getCore() { 
            return static_cast<LogShear &>(*_core); 
        }   
    
        Ellipse * clone() const { return new Ellipse(*this); }

        void setComplex(std::complex<double> const & gamma) { 
            getCore().setComplex(gamma); 
        }
        std::complex<double> getComplex() const { 
            return getCore().getComplex(); 
        }
        void setGamma(double gamma) { getCore().setGamma(gamma); }
        double getGamma() const { return getCore().getGamma(); }

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
            _core.reset(new LogShear(vector.segment<3>(2)));
        }

        explicit Ellipse(
            LogShear const & core, 
            Coordinate const & center = Coordinate(0,0)
        ) : Super(core,center) {}

        Ellipse(
            Super const & other
        ) : Super(new LogShear(other.getCore()), other.getCenter()) {}

        Ellipse(
            Ellipse const & other
        ) : Super(new LogShear(other.getCore()), other.getCenter()) {}
    };

    
    virtual std::string const getName() const { return "LogShear"; }

    virtual void scale(double ratio) { _vector[KAPPA] += std::log(ratio); }

    virtual Eigen::Matrix3d getScalingDerivative(double ratio) const {
        return Eigen::Matrix3d::Identity();
    }

    virtual bool normalize() { return true; }

    virtual LogShear * clone() const { return new LogShear(*this); }

    virtual Ellipse * makeEllipse(
        Coordinate const & center = Coordinate(0,0)
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
