// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_GEOM_ELLIPSES_Convolution_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Convolution_h_INCLUDED

/**
 *  @file
 *  @brief Definitions for BaseEllipse::Convolution and EllipseCore::Convolution.
 *
 *  @note Do not include directly; use the main ellipse header file.
 */

#include <boost/tuple/tuple.hpp>

#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief A temporary-only expression object for ellipse core convolution.
 */
class EllipseCore::Convolution {
public:

    /// Matrix type for derivative with respect to input ellipse parameters.
    typedef Eigen::Matrix3d DerivativeMatrix; 

    /// @brief Standard constructor.
    Convolution(EllipseCore & self, EllipseCore const & other) :
        self(self), other(other) {}

    /// @brief Return a new convolved ellipse core.
    PTR(EllipseCore) copy() const;

    /// @brief Convolve the ellipse core in-place.
    void inPlace();

    /// @brief Return the derivative of convolved core with respect to self.
    DerivativeMatrix d() const;
    
    void apply(EllipseCore & result) const;
 
    EllipseCore & self;
    EllipseCore const & other;

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
    Ellipse copy() const;

    /// @brief Convolve the ellipse in-place.
    void inPlace();

    /// @brief Return the derivative of convolved ellipse with respect to self.
    DerivativeMatrix d() const;

    Ellipse & self;
    Ellipse const & other;

};

inline EllipseCore::Convolution EllipseCore::convolve(EllipseCore const & other) {
    return EllipseCore::Convolution(*this, other);
}

inline EllipseCore::Convolution const EllipseCore::convolve(EllipseCore const & other) const {
    return EllipseCore::Convolution(const_cast<EllipseCore &>(*this), other);
}

inline Ellipse::Convolution Ellipse::convolve(Ellipse const & other) {
    return Ellipse::Convolution(*this, other);
}

inline Ellipse::Convolution const Ellipse::convolve(Ellipse const & other) const {
    return Ellipse::Convolution(const_cast<Ellipse &>(*this), other);
}

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Convolution_h_INCLUDED
