// -*- lsst-c++ -*-
#ifndef LSST_AFW_GEOM_ELLIPSES_ELLIPSEIMPL_H
#define LSST_AFW_GEOM_ELLIPSES_ELLIPSEIMPL_H

/**
 *  \file
 *  \brief Definitions for EllipseImpl and CoreImpl.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseEllipse.h"

namespace lsst {
namespace afw {
namespace geom { 
namespace ellipses {
namespace detail {

/**
 *  \internal \brief A CRTP base class for concrete Ellipse subclasses.
 *
 *  EllipseImpl provides operations whose implementations are common across all subclasses but cannot
 *  be moved to the base class.
 *
 *  \ingroup EllipseGroup
 */
template <typename DerivedCore, typename DerivedEllipse>
class EllipseImpl : public BaseEllipse {
    class Transformer;
public:

    typedef boost::shared_ptr<DerivedEllipse> Ptr;
    typedef boost::shared_ptr<DerivedEllipse const> ConstPtr;

    typedef DerivedEllipse Ellipse;
    typedef DerivedCore Core;

    /// \brief Deep-copy the ellipse.
    boost::shared_ptr<DerivedEllipse> clone() const {
        return boost::shared_ptr<DerivedEllipse>(static_cast<DerivedEllipse*>(_clone())); 
    }

    /// \brief Return the Core object.
    DerivedCore const & getCore() const { return static_cast<DerivedCore const &>(*_core); }

    /// \brief Return the Core object.
    DerivedCore & getCore() { return static_cast<DerivedCore &>(*_core); }

    /// \brief Transform the ellipse by the given AffineTransform.
    Transformer transform(AffineTransform const & transform);
    Transformer const transform(AffineTransform const & transform) const; ///< \copybrief transform

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    DerivedEllipse & operator=(BaseEllipse const & other) {
        return static_cast<DerivedEllipse &>(BaseEllipse::operator=(other));
    }

    /// \brief Set the parameters of this ellipse from another.
    DerivedEllipse & operator=(DerivedEllipse const & other) {
        if (&other != static_cast<DerivedEllipse*>(this)) {
            _center = other.getCenter();
            _core->setVector(other.getCore().getVector());
        }
        return static_cast<DerivedEllipse&>(*this);
    }

protected:

    virtual BaseEllipse * _clone() const {
        return new DerivedEllipse(static_cast<DerivedEllipse const &>(*this)); 
    }

    explicit EllipseImpl(BaseCore const & core, PointD const & center) : BaseEllipse(core,center) {}

    explicit EllipseImpl(BaseCore * core, PointD const & center) : BaseEllipse(core,center) {}

    explicit EllipseImpl(BaseEllipse::ParameterVector const & vector) : 
        BaseEllipse(new DerivedCore(vector.segment<3>(2)), PointD(vector.segment<2>(0))) {}
};

/**
 *  \internal \brief A CRTP base class for concrete Core subclasses.
 *
 *  CoreImpl provides operations whose implementations are common across all subclasses but cannot
 *  be moved to the base class.
 *
 *  \ingroup EllipseGroup
 */
template <typename DerivedCore, typename DerivedEllipse>
class CoreImpl : public BaseCore {
    class Transformer;
public:
    
    typedef boost::shared_ptr<DerivedCore> Ptr;
    typedef boost::shared_ptr<const DerivedCore> ConstPtr;

    typedef DerivedEllipse Ellipse;
    typedef DerivedCore Core;

    /// \brief Deep copy the ellipse core.
    boost::shared_ptr<DerivedCore> clone() const {
        return boost::shared_ptr<DerivedCore>(static_cast<DerivedCore*>(_clone())); 
    }

    /// \brief Construct an Ellipse of the appropriate subclass from this and the given center.
    boost::shared_ptr<DerivedEllipse> makeEllipse(PointD const & center = PointD()) const {
        return boost::shared_ptr<DerivedEllipse>(static_cast<DerivedEllipse*>(_makeEllipse(center)));
    }

    /// \brief Transform the ellipse core by the given AffineTransform.
    Transformer transform(AffineTransform const & transform);
    Transformer const transform(AffineTransform const & transform) const; ///< \copybrief transform
    
    /// \brief Assign other to this and return the derivative of the conversion, d(this)/d(other).
    virtual Jacobian dAssign(BaseCore const & other) {
        return other._dAssignTo(static_cast<DerivedCore &>(*this)); 
    }

protected:

    virtual BaseCore * _clone() const {
        return new DerivedCore(static_cast<DerivedCore const &>(*this)); 
    }
    
    virtual BaseEllipse * _makeEllipse(PointD const & center) const {
        return new DerivedEllipse(static_cast<DerivedCore const &>(*this), center);
    }

    explicit CoreImpl(ParameterVector const & vector) : BaseCore(vector) {}

    explicit CoreImpl(double v1=0, double v2=0, double v3=0) : BaseCore(v1,v2,v3) {}

};

} // namespace lsst::afw::geom::ellipses::detail
} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_ELLIPSEIMPL_H
