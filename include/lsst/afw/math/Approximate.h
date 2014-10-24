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
template<typename T> class Approximate;

 /**
  * @brief Control how to make an approximation
  *
  * \note the x- and y-order must be the same, due to a limitation of Chebyshev1Function2
  *
  * @ingroup afw
  */
class ApproximateControl {
public:
    /// \brief Choose the type of approximation to use
    enum Style {
        UNKNOWN = -1,
        CHEBYSHEV,                      ///< Use a 2-D Chebyshev polynomial
        NUM_STYLES
    };

    ApproximateControl(Style style, int orderX, int orderY=-1);
    /// Return the Style
    Style getStyle() const { return _style; }
    void setStyle(Style const style) { _style = style; }
    /// Return the order of approximation to use in the x-direction
    int getOrderX() const { return _orderX; }
    void setOrderX(int const orderX) { _orderX = orderX; }
    /// Return the order of approximation to use in the y-direction
    int getOrderY() const { return _orderY; }
    void setOrderY(int const orderY) { _orderY = orderY; }
private:
    Style _style;
    int _orderX;
    int _orderY;
};

/**
 *  @brief Construct a new Approximate object, inferring the type from the type of the given MaskedImage.
 */
template<typename PixelT>
PTR(Approximate<PixelT>)
makeApproximate(std::vector<double> const &x, std::vector<double> const &y,
                image::MaskedImage<PixelT> const& im, geom::Box2I const& bbox,
                ApproximateControl const& ctrl);

/**
 * @brief Approximate values for a MaskedImage
 * @ingroup afw
 */
template<typename PixelT>
class Approximate {
public:
    typedef float OutPixelT;            ///< The pixel type of returned images

    friend PTR(Approximate<PixelT>)
    makeApproximate<>(std::vector<double> const &x, std::vector<double> const &y,
                    image::MaskedImage<PixelT> const& im, geom::Box2I const& bbox,
                    ApproximateControl const& ctrl);
    /// \brief dtor
    virtual ~Approximate() {}
    /// \brief Return the approximate %image as a Image
    PTR(image::Image<OutPixelT>) getImage(int orderX=-1, int orderY=-1) const {
        return doGetImage(orderX, orderY);
    }
    /// \brief Return the approximate %image as a MaskedImage
    PTR(image::MaskedImage<OutPixelT>) getMaskedImage(int orderX=-1, int orderY=-1) const {
        return doGetMaskedImage(orderX, orderY);
    }
protected:
    /**
     * \brief Base class ctor
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
    virtual PTR(image::Image<OutPixelT>) doGetImage(int orderX, int orderY) const = 0;
    virtual PTR(image::MaskedImage<OutPixelT>) doGetMaskedImage(int orderX, int orderY) const = 0;
};

}}}
                     
#endif // LSST_AFW_MATH_APPROXIMATE_H
