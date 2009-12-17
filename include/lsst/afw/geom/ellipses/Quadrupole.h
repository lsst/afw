// -*- lsst-c++ -*-
#ifndef LSST_AFW_GEOM_ELLIPSES_QUADRUPOLE_H
#define LSST_AFW_GEOM_ELLIPSES_QUADRUPOLE_H

/**
 *  \file
 *  \brief Definitions for Quadrupole and QuadrupoleEllipse.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/EllipseImpl.h"

namespace lsst {
namespace afw {
namespace geom { 
namespace ellipses {

class QuadrupoleEllipse;

/**
 *  \brief An ellipse core using the quadrupole moments of an elliptical Gaussian (ixx,iyy,ixy) 
 *         as its parameters.
 *
 *  \ingroup EllipseGroup
 */
class Quadrupole : public detail::CoreImpl<Quadrupole,QuadrupoleEllipse> {
    typedef detail::CoreImpl<Quadrupole,QuadrupoleEllipse> Super;
public:

    typedef Eigen::Matrix2d Matrix; ///< Matrix type for the matrix representation of Quadrupole parameters.

    enum Parameters { IXX=0, IYY=1, IXY=2 }; ///< Definitions for elements of a core vector.

    /// Return a string that identifies this parametrization.
    virtual char const * getName() const { return "Quadrupole"; }

    /**
     *  \brief Put the parameters into a "standard form", if possible, and return
     *         false if they are entirely invalid.
     */
    virtual bool normalize() { return getDeterminant() >= 0; }

    /// \brief Increase the major and minor radii of the ellipse core by the given buffer.
    virtual void scale(double ratio) { _vector *= (ratio*ratio); }

    /// \brief Return a 2x2 symmetric matrix of the parameters.
    Matrix getMatrix() const {
        Matrix r; r << _vector[IXX], _vector[IXY], _vector[IXY], _vector[IYY]; return r;
    }

    /// \brief Return the determinant of the matrix representation.
    double getDeterminant() const { return _vector[IXX]*_vector[IYY] - _vector[IXY]*_vector[IXY]; }

    /// \brief Standard assignment.
    Quadrupole & operator=(Quadrupole const & other) { _vector = other._vector; return *this; }

    /// \brief Converting assignment.
    virtual Quadrupole & operator=(BaseCore const & other) { other._assignTo(*this); return *this; }

    /// \brief Construct from a parameter vector.
    explicit Quadrupole(BaseCore::ParameterVector const & vector) : Super(vector) {}

    /// \brief Construct from parameter values.
    explicit Quadrupole(double xx=0, double yy=0, double xy=0) : Super(xx,yy,xy) {}
    
    /// \brief Converting copy constructor.
    Quadrupole(BaseCore const & other) { *this = other; }

    /// \brief Copy constructor.
    Quadrupole(Quadrupole const & other) : Super(other.getVector()) {}

protected:
    
    virtual void _assignTo(Quadrupole & other) const;
    virtual void _assignTo(Axes & other) const;
    virtual void _assignTo(Distortion & other) const;
    virtual void _assignTo(LogShear & other) const;

    virtual Jacobian _dAssignTo(Quadrupole & other) const;
    virtual Jacobian _dAssignTo(Axes & other) const;
    virtual Jacobian _dAssignTo(Distortion & other) const;
    virtual Jacobian _dAssignTo(LogShear & other) const;

};

/**
 *  \brief An ellipse with a Quadrupole core.
 *
 *  \ingroup EllipseGroup
 */
class QuadrupoleEllipse : public detail::EllipseImpl<Quadrupole,QuadrupoleEllipse> {
    typedef detail::EllipseImpl<Quadrupole,QuadrupoleEllipse> Super;
public:

    /// Definitions for elements of an ellipse vector.
    enum Parameters { X=0, Y=1, IXX=2, IYY=3, IXY=4 };

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    QuadrupoleEllipse & operator=(BaseEllipse const & other) { return Super::operator=(other); }

    /// \brief Set the parameters of this ellipse from another.
    QuadrupoleEllipse & operator=(QuadrupoleEllipse const & other) { return Super::operator=(other); }

    /// \brief Construct from a PointD and zero-size Core.
    explicit QuadrupoleEllipse(PointD const & center = PointD()) : Super(new Quadrupole(),center) {}

    /// \brief Construct from a copy of a Quadrupole core.
    explicit QuadrupoleEllipse(Quadrupole const & core, PointD const & center = PointD()) : 
        Super(core,center) {}

    /// \brief Construct from a 5-element parameter vector.
    explicit QuadrupoleEllipse(BaseEllipse::ParameterVector const & vector) : Super(vector) {}

    /// \brief Converting copy constructor.
    QuadrupoleEllipse(BaseEllipse const & other) : Super(other.getCore(),other.getCenter()) {}

    /// \brief Copy constructor.
    QuadrupoleEllipse(QuadrupoleEllipse const & other) : Super(other.getCore(),other.getCenter()) {}

};

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_QUADRUPOLE_H
