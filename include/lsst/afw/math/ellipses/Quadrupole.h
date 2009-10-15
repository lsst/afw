#ifndef LSST_AFW_MATH_ELLIPSES_QUADRUPOLE_H
#define LSST_AFW_MATH_ELLIPSES_QUADRUPOLE_H

#include <lsst/afw/math/ellipses/Core.h>
#include <lsst/afw/math/ellipses/Ellipse.h>
#include <Eigen/LU>

namespace lsst { 
namespace afw {
namespace math {
namespace ellipses {

typedef Eigen::Matrix2d QuadrupoleMatrix;


class Quadrupole : public Core {
public:
    typedef boost::shared_ptr<Quadrupole> Ptr;
    typedef boost::shared_ptr<const Quadrupole> ConstPtr;

    enum Parameters { IXX=0, IYY=1, IXY=2 };

    class Ellipse : public lsst::afw::math::ellipses::Ellipse {
        typedef lsst::afw::math::ellipses::Ellipse Super;
    public: 
        typedef boost::shared_ptr<Ellipse> Ptr;
        typedef boost::shared_ptr<const Ellipse> ConstPtr;
    
        enum Parameters { 
            X=Super::X, Y=Super::Y, 
            IXX=Quadrupole::IXX+2, 
            IYY=Quadrupole::IYY+2, 
            IXY=Quadrupole::IXY+2
        };

        Quadrupole const & getCore() const { return static_cast<Quadrupole const &>(*_core); }
        Quadrupole & getCore() { return static_cast<Quadrupole &>(*_core); }

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
            _core.reset(new Quadrupole(vector.segment<3>(2)));
        }
    
        explicit Ellipse(
            Quadrupole const & core, 
            Coordinate const & center = Coordinate(0,0)
        ) : Super(core,center) {}

        Ellipse(Super const & other) 
            : Super(new Quadrupole(other.getCore()), other.getCenter()) {}

        Ellipse(Ellipse const & other) 
            : Super(new Quadrupole(other.getCore()), other.getCenter()) {}
    };

    virtual std::string const getName() const { return "Quadrupole"; }

    virtual void scale(double ratio) { _vector *= (ratio*ratio); }

    virtual Eigen::Matrix3d getScalingDerivative(double ratio) const {
        Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
        r(0,0) = r(1,1) = r(2,2) = ratio*ratio;
        return r;
    }

    virtual void transform(AffineTransform const & transform);

    virtual bool normalize() { return getDeterminant() >= 0; }

    virtual Quadrupole * clone() const { return new Quadrupole(*this); }

    virtual Ellipse * makeEllipse(
        Coordinate const & center = Coordinate(0,0)
    ) const;

    double getDeterminant() const { 
        return _vector[IXX]*_vector[IYY] - _vector[IXY]*_vector[IXY]; 
    }

    QuadrupoleMatrix getMatrix() const {
        QuadrupoleMatrix r; 
        r << _vector[IXX], _vector[IXY], _vector[IXY], _vector[IYY]; 
        return r;
    }

    template <typename Derived>
    explicit Quadrupole(Eigen::MatrixBase<Derived> const & vector) 
        : Core(vector) {}

    explicit Quadrupole(double xx=0, double yy=0, double xy=0) 
        : Core(xx,yy,xy) {}
    
    Quadrupole(Core const & other) { *this = other; }

    Quadrupole & operator=(Quadrupole const & other) { 
        _vector = other._vector; 
        return *this; 
    }
    virtual Quadrupole & operator=(Core const & other) { 
        other.assignTo(*this); 
        return *this; 
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

#endif // !LSST_AFW_MATH_ELLIPSES_QUADRUPOLE_H
