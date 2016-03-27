// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_GEOM_ELLIPSES_Convolution_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Convolution_h_INCLUDED

/**
 *  @file
 *  @brief Definitions for BaseEllipse::Convolution and BaseCore::Convolution.
 *
 *  @note Do not include directly; use the main ellipse header file.
 */

#include <boost/tuple/tuple.hpp>

#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief A temporary-only expression object for ellipse core convolution.
 */
class BaseCore::Convolution {
public:

    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix3d DerivativeMatrix; 

    /// @brief Standard constructor.
    Convolution(BaseCore & self, BaseCore const & other) :
        self(self), other(other) {}

    /// @brief Return a new convolved ellipse core.
    BaseCore::Ptr copy() const;

    /// @brief Convolve the ellipse core in-place.
    void inPlace();

    /// @brief Return the derivative of convolved core with respect to self.
    DerivativeMatrix d() const;
    
    void apply(BaseCore & result) const;
 
    BaseCore & self;
    BaseCore const & other;

};

/**
 *  @brief A temporary-only expression object for ellipse convolution.
 */
class Ellipse::Convolution {
public:

    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix<double,5,5> DerivativeMatrix; 

    /// @brief Standard constructor.
    Convolution(Ellipse & self, Ellipse const & other) :
        self(self), other(other) {}

    /// @brief Return a new convolved ellipse.
    Ellipse::Ptr copy() const;

    /// @brief Convolve the ellipse in-place.
    void inPlace();

    /// @brief Return the derivative of convolved ellipse with respect to self.
    DerivativeMatrix d() const;

    Ellipse & self;
    Ellipse const & other;

};

inline BaseCore::Convolution BaseCore::convolve(BaseCore const & other) {
    return BaseCore::Convolution(*this, other);
}

inline BaseCore::Convolution const BaseCore::convolve(BaseCore const & other) const {
    return BaseCore::Convolution(const_cast<BaseCore &>(*this), other);
}

inline Ellipse::Convolution Ellipse::convolve(Ellipse const & other) {
    return Ellipse::Convolution(*this, other);
}

inline Ellipse::Convolution const Ellipse::convolve(Ellipse const & other) const {
    return Ellipse::Convolution(const_cast<Ellipse &>(*this), other);
}

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Convolution_h_INCLUDED
