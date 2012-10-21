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
namespace geom {
    class Box2I;
}
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

    ApproximateControl(Style style, int orderX, int orderY=0);

    Style getStyle() const { return _style; }
    int getOrderX() const { return _orderX; }
    int getOrderY() const { return _orderY; }
private:
    Style const _style;
    int const _orderX;
    int const _orderY;
};

/**
 * @brief Approximate values for a set of x,y vector<>s
 * @ingroup afw
 */
template<typename PixelT>
class Approximate {
public:
    typedef float OutPixelT;

    friend PTR(Approximate<PixelT>)
    makeApproximate(std::vector<double> const &x, std::vector<double> const &y,
                    image::MaskedImage<PixelT> const& im, geom::Box2I const& bbox,
                    ApproximateControl const& ctrl);
    
    virtual ~Approximate() {}

    PTR(image::MaskedImage<OutPixelT>) getImage(bool const getMaskedImage=true) const {
        return doGetImage(getMaskedImage);
    }
protected:
    /**
     * Base class ctor
     */
    Approximate(std::vector<double> const &x,         ///< the x-values of points
                std::vector<double> const &y,         ///< the y-values of points
                geom::Box2I const& bbox,              ///< Range where approximation should be valid
                ApproximateControl const& ctrl        ///< desired approximation algorithm
               ) : _xVec(x), _yVec(y), _bbox(bbox), _ctrl(ctrl) {}

    std::vector<double> const _xVec;    ///< the x-values of points
    std::vector<double> const _yVec;    ///< the y-values of points
    geom::Box2I const _bbox;            ///< Domain for approximation
    ApproximateControl const _ctrl;     ///< desired approximation algorithm
private:
    Approximate(Approximate const&);
    Approximate& operator=(Approximate const&);
    virtual PTR(image::MaskedImage<OutPixelT>) doGetImage(bool const getMaskedImage) const = 0;
};

template<typename PixelT>
PTR(Approximate<PixelT>)
makeApproximate(std::vector<double> const &x, std::vector<double> const &y,
                image::MaskedImage<PixelT> const& im, geom::Box2I const& bbox,
                ApproximateControl const& ctrl);

}}}
                     
#endif // LSST_AFW_MATH_APPROXIMATE_H
