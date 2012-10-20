// -*- LSST-C++ -*-
#if !defined(LSST_AFW_MATH_APPROXIMATE_H)
#define LSST_AFW_MATH_APPROXIMATE_H

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
#include "lsst/base.h"

namespace lsst {
namespace afw {
namespace image {
template<typename PixelT> class Image;
template<typename PixelT, typename U, typename V> class MaskedImage;
}
namespace math {
 /**
  * @brief Approximate values for a set of x,y vector<>s
  * @ingroup afw
  */
template<typename T> class Approximate;

class ApproximateControl {
public:
    enum Style {
        UNKNOWN = -1,
        CHEBYSHEV,
        NUM_STYLES
    };
    template<typename T> friend class Approximate;

    ApproximateControl(Style style, int orderX, int orderY) :
        _style(style), _orderX(orderX), _orderY(orderY) {}
private:
    Style _style;
    int _orderX;
    int _orderY;
};

/**
 * @brief Approximate values for a set of x,y vector<>s
 * @ingroup afw
 */
template<typename PixelT>
class Approximate {
public:
    friend PTR(Approximate<PixelT>)
    makeApproximate(std::vector<double> const &x, std::vector<double> const &y,
                    image::MaskedImage<PixelT> const& im,
                    ApproximateControl::Style const& style);
    
    virtual ~Approximate() {}
    virtual double approximate(double const x, double const y) const = 0;
protected:
    /**
     * Base class ctor
     */
    Approximate(std::vector<double> const &x,         ///< the x-values of points
                std::vector<double> const &y,         ///< the y-values of points
                ApproximateControl::Style const& style ///< desired approximation algorithm
               ) : _x(x), _y(y), _style(style) {}

    std::vector<double> const _x;           ///< the x-values of points
    std::vector<double> const _y;           ///< the y-values of points
    ApproximateControl::Style const _style; ///< desired approximation algorithm
private:
    Approximate(Approximate const&);
    Approximate& operator=(Approximate const&);
};

template<typename PixelT>
PTR(Approximate<PixelT>)
makeApproximate(std::vector<double> const &x, std::vector<double> const &y,
                image::MaskedImage<PixelT> const& im, ApproximateControl::Style const& style);

}}}
                     
#endif // LSST_AFW_MATH_APPROXIMATE_H
