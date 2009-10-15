#ifndef LSST_AFW_MATH_ELLIPSES_CORE_H
#define LSST_AFW_MATH_ELLIPSES_CORE_H

#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <Eigen/Core>

#include <lsst/afw/math/AffineTransform.h>

namespace lsst {
namespace afw { 
namespace math {
namespace ellipses {

//forward declaration of subclasses of class Core
class Quadrupole;
class Axes;
class Distortion;
class LogShear;

//forward declaration of class Ellipse
class Ellipse;

class Core {
public:

    class TransformDerivative {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
        typedef boost::shared_ptr<TransformDerivative> Ptr;
        typedef boost::shared_ptr<const TransformDerivative> ConstPtr;

        TransformDerivative(
            Core const & core, 
            AffineTransform const & transform
        );

        // Derivative of transform output core with respect to input core.
        Eigen::Matrix3d dInput() const;

        // Derivative of transform output core with respect to 
        //transform parameters.
        Eigen::Matrix<double,3,6> dTransform() const;

        AffineTransform & transform() {return _transform;}
        AffineTransform const & transform() const {return _transform;}

    private:
        AffineTransform _transform;
        Eigen::Vector3d _quadrupole;
        Eigen::Matrix3d _jacobian;
        Eigen::Matrix3d _jacobian_inv;
    };


    typedef boost::shared_ptr<Core> Ptr;
    typedef boost::shared_ptr<const Core> ConstPtr;

    virtual std::string const getName() const = 0;

    virtual void scale(double ratio) = 0;

    // Acale the core by the given ratio, and return the derivative of the
    // scaled parameters with respect to unscaled parameters.
    virtual Eigen::Matrix3d getScalingDerivative(double ratio) const = 0;

    // default implementation converts to Quadrupole and back.
    virtual void transform(AffineTransform const & transform);

    // put the parameters into a "standard form", if possible, and return
    // false if they are entirely invalid.
    virtual bool normalize() = 0;
    
    // default implementation converts to Axes and back.
    virtual AffineTransform getGenerator() const;

    double & operator[](int i) { return _vector[i]; }
    double const operator[](int i) const { return _vector[i]; }

    Eigen::Vector3d const getVector() const { return _vector; }

    template <typename Derived>
    void setVector(Eigen::MatrixBase<Derived> const & vector) { _vector = vector; }

    virtual Core & operator=(Core const & other) = 0;
    
    // Assign other to this and return the
    // derivative of the conversion, d(this)/d(other).
    virtual Eigen::Matrix3d differentialAssign(Core const & other) = 0;

    virtual Core * clone() const = 0;

    virtual Ellipse * makeEllipse(Coordinate const & center = Coordinate(0,0)) const = 0;

    virtual ~Core() {}

protected:

    template <typename Derived>
    explicit Core(Eigen::MatrixBase<Derived> const & vector) : _vector(vector) {}

    explicit Core(double v1=0, double v2=0, double v3=0) : _vector(v1,v2,v3) {}

    virtual void assignTo(Quadrupole & other) const = 0;
    virtual void assignTo(Axes & other) const = 0;
    virtual void assignTo(Distortion & other) const = 0;
    virtual void assignTo(LogShear & other) const = 0;

    virtual Eigen::Matrix3d differentialAssignTo(Quadrupole & other) const = 0;
    virtual Eigen::Matrix3d differentialAssignTo(Axes & other) const = 0;
    virtual Eigen::Matrix3d differentialAssignTo(Distortion & other) const = 0;
    virtual Eigen::Matrix3d differentialAssignTo(LogShear & other) const = 0;

    Eigen::Vector3d _vector;

    friend class Quadrupole;
    friend class Axes;
    friend class Distortion;
    friend class LogShear;
};

}}}} //end namespace lsst::afw::math::ellipses

#endif // LSST_AFW_MATH_ELLIPSES_CORE_H

