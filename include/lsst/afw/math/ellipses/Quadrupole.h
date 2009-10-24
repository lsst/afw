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

class Quadrupole;

class QuadrupoleEllipse : public Ellipse {
public: 
    typedef boost::shared_ptr<QuadrupoleEllipse> Ptr;
    typedef boost::shared_ptr<const QuadrupoleEllipse> ConstPtr;
     
    enum Parameters {X=0, Y,IXX, IYY, IXY};

    Quadrupole const & getCore() const;
    Quadrupole & getCore();

    QuadrupoleEllipse * clone() const { return new QuadrupoleEllipse(*this); }

    QuadrupoleEllipse & operator=(Ellipse const & other) {
        return static_cast<QuadrupoleEllipse &>(Ellipse::operator=(other));
    }

    explicit QuadrupoleEllipse(Coordinate const & center);

    template <typename Derived>
    explicit QuadrupoleEllipse(Eigen::MatrixBase<Derived> const & vector); 
    
    explicit QuadrupoleEllipse(
        Quadrupole const & core, 
        lsst::afw::math::Coordinate const & center = Coordinate(0,0)
    );

    QuadrupoleEllipse(Ellipse const & other);

    QuadrupoleEllipse(QuadrupoleEllipse const & other) ;
};

class Quadrupole : public Core {
public:
    typedef boost::shared_ptr<Quadrupole> Ptr;
    typedef boost::shared_ptr<const Quadrupole> ConstPtr;
    typedef QuadrupoleEllipse Ellipse;

    enum Parameters { IXX=0, IYY=1, IXY=2 };

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
