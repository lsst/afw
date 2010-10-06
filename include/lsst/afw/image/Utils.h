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
 
/**
 * \file
 * \brief A set of classes of general utility in connection with images
 * 
 * We provide representations of points, bounding boxes, circles etc.
 */
#ifndef LSST_AFW_IMAGE_UTILS_H
#define LSST_AFW_IMAGE_UTILS_H

#include <list>
#include <map>
#include <string>
#include <utility>

#include "boost/format.hpp"
#include "boost/mpl/bool.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/afw/image/lsstGil.h"
#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/formatters/ImageFormatter.h"

namespace lsst { namespace afw { namespace image {
    /************************************************************************************************************/
    /**
     * \brief a single Point, templated over the type to be used to store the coordinates
     * 
     * \note Access to the coordinates is provided via get/set functions --- is this
     * a case where direct access would be appropriate (cf. \c std::pair)?
     */
    template<typename T>
    class Point {
    public:
        Point(T val=0                   ///< The Point is (val, val)
             ) : _x(val), _y(val) {}
        Point(T x,                      ///< x-value
              T y                       ///< y-value
             ) : _x(x), _y(y) {}
        Point(T const xy[2]) : _x(xy[0]), _y(xy[1]) {}

        T getX() const { return _x; }   ///< Return x coordinate
        T getY() const { return _y; }   ///< Return y coordinate
        void setX(T x) { _x = x; }      ///< Set x coordinate
        void setY(T y) { _y = y; }      ///< Set y coordinate

        bool operator==(const Point& rhs) const { return (_x == rhs._x && _y == rhs._y); }
        bool operator!=(const Point& rhs) const { return !(*this == rhs); }
        
        Point operator+(const Point& rhs) const { return Point(_x + rhs._x, _y + rhs._y); }
        Point operator-(const Point& rhs) const { return Point(_x - rhs._x, _y - rhs._y); }

        //! Offset a Point by the specified vector
        Point& shift(T dx,                ///< How much to offset in x
                     T dy                 ///< How much to offset in y
                    ) {
            *this = *this + Point(dx, dy);
            return *this;
        }
#if !defined(SWIG)
        /// Return x (i == 0) or y (i == 1)
        T const& operator[](int const i) const {
            switch (i) {
              case 0: return _x;
              case 1: return _y;
              default: throw LSST_EXCEPT(lsst::pex::exceptions::RangeErrorException, (boost::format("Index i == %d must be 0 or 1") % i).str());
            }
        }
        /// Return x (i == 0) or y (i == 1)
        T& operator[](int const i) {
            return const_cast<T&>((static_cast<const Point&>(*this))[i]); // Meyers, Effective C++, Item 3
        }
#endif
    private:
        T _x, _y;
    };
    typedef Point<double> PointD;
    typedef Point<int> PointI;

    /**
     * @brief A bounding box, i.e. a rectangular region defined by its corners, or origin and dimensions
     *
     * Note that the corners are interpreted as being included in the box, so
     * <tt>BBox(PointI(1, 1), PointI(2, 3))</tt> has width 2 and height 3
     */
    class BBox : private std::pair<PointI, std::pair<int, int> > {
    public:
        //! Create a BBox with origin llc and the specified dimensions
        BBox() :
            std::pair<PointI, std::pair<int, int> >(PointI(0,0), std::pair<int, int>(0, 0)) {} 

        BBox(PointI llc,                ///< Desired lower left corner
             int width=1,               ///< Width of BBox (pixels)
             int height=1               ///< Height of BBox (pixels)
            ) :
            std::pair<PointI, std::pair<int, int> >(llc, std::pair<int, int>(width, height)) {} 
        //! Create a BBox given two corners
        BBox(PointI llc,                ///< Desired lower left corner
             PointI urc                 ///< Desired upper right corner
            ) :
            std::pair<PointI, std::pair<int, int> >(llc,
                                                    std::pair<int, int>(urc.getX() - llc.getX() + 1,
                                                                        urc.getY() - llc.getY() + 1)) {}

        //! Return true iff the point lies in the BBox
        bool contains(PointI p          ///< The point to check
                     ) const {
            return p.getX() >= getX0() && p.getX() <= getX1() && p.getY() >= getY0() && p.getY() <= getY1();
        }

        //! Grow the BBox to include the specified PointI
        void grow(PointI p              ///< The point to include
                 ) {
            if (getWidth() == 0 && getHeight() == 0) {
                first.setX(p.getX());
                first.setY(p.getY());
                second.first = 1;
                second.second = 1;

                return;
            }

            if (p.getX() < getX0()) {
                second.first = getWidth() + (getX0() - p.getX());
                first.setX(p.getX());
            } else if (p.getX() > getX1()) {
                second.first = p.getX() - getX0() + 1;
            }

            if (p.getY() < getY0()) {
                second.second = getHeight() + (getY0() - p.getY());
                first.setY(p.getY());
            } else if (p.getY() > getY1()) {
                second.second = p.getY() - getY0() + 1;
            }
        }
        //! Offset a BBox by the specified vector
        void shift(int dx,               ///< How much to offset in x
                   int dy                ///< How much to offset in y
                  ) {
            first.setX(first.getX() + dx);
            first.setY(first.getY() + dy);
            second = second;
            }

        int getX0() const { return first.getX(); } ///< Return x coordinate of lower-left corner
        /// Set x coordinate of lower-left corner
        void setX0(int x0) {
            second.first += (first.getX() - x0);
            first.setX(x0);
        }
        int getY0() const { return first.getY(); } ///< Return y coordinate of lower-left corner
        /// Set y coordinate of lower-left corner
        void setY0(int y0) {
            second.second += (first.getY() - y0);
            first.setY(y0);
        }
        int getX1() const { return first.getX() + second.first - 1; } ///< Return x coordinate of upper-right corner
        void setX1(int x1) { second.first = x1 - getX0() + 1; } ///< Set x coordinate of lower-left corner
        int getY1() const { return first.getY() + second.second - 1; } ///< Return y coordinate of upper-right corner
        void setY1(int y1) { second.second = y1 - getY0() + 1; } ///< Set y coordinate of lower-left corner
        PointI getLLC() const { return first; } ///< Return lower-left corner
        PointI getURC() const { return PointI(getX1(), getY1()); } ///< Return upper-right corner
        int getWidth() const { return second.first; } ///< Return width of BBox (== <tt>X1 - X0 + 1</tt>)
        int getHeight() const { return second.second; } ///< Return height of BBox (== <tt>Y1 - Y0 + 1</tt>)
        /// Set BBox's width
        void setWidth(int width         ///< Desired width
                     ) { second.first = width; } 
        /// Set BBox's height
        void setHeight(int height       ///< Desired height
                     ) { second.second = height; } 
        const std::pair<int, int> getDimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }

        /// Clip this with rhs
        void clip(BBox const&rhs) {
            if (rhs.getX1() < getX1()) { setX1(rhs.getX1()); } // do the width, which is set from (x0, y0), first
            if (rhs.getY1() < getY1()) { setY1(rhs.getY1()); }

            if (rhs.getX0() > getX0()) { setX0(rhs.getX0()); }
            if (rhs.getY0() > getY0()) { setY0(rhs.getY0()); }

            if (getWidth() < 0 || getHeight() < 0) {
                setWidth(0);
                setHeight(0);
            }
        }

        operator bool() const {
            return !(getWidth() == 0 && getHeight() == 0);
        }

        bool operator==(const BBox& rhs) const {
            return
                getX0() == rhs.getX0() && getY0() == rhs.getY0() &&
                getWidth() == rhs.getWidth() && getHeight() == rhs.getHeight();
        }
        bool operator!=(const BBox& rhs) const {
            return !operator==(rhs);
        }
    };

    /**
     * \brief A BCircle (named by analogy to BBox) is used to describe a circular patch of pixels
     *
     * \note Only integer centers are supported.  It'd be easy enough to add other
     * types, but as BCircle is designed by analogy to BBox (i.e. to define sets of pixels),
     * I haven't done so.
     */
    class BCircle : private std::pair<PointI, float > {
    public:
        /// Create a BCircle given centre and radius
        BCircle(PointI center,               //!< Centre of circle
                float r                      //!< Radius of circle
               ) : std::pair<PointI, float>(center, fabs(r)) {}

        PointI const& getCenter() const { return first; } ///< Return the circle's centre
        float getRadius() const { return second; }        ///< Return the circle's radius
        BBox getBBox() const {                           ///< Return the circle's bounding box
            int const iradius = static_cast<int>(second + 0.5);
            PointI llc(first[0] - iradius, first[1] - iradius);
            PointI urc(first[0] + iradius, first[1] + iradius);
            return BBox(llc, urc);
        }
    };

/************************************************************************************************************/

lsst::daf::base::PropertySet::Ptr readMetadata(std::string const& fileName, const int hdu=0, bool strip=false);
lsst::daf::base::PropertySet::Ptr readMetadata(char **ramFile, size_t *ramFileLen, const int hdu=0, bool strip=false);

/************************************************************************************************************/
/**
 * Return a value indicating a bad pixel for the given Image type
 *
 * A quiet NaN is returned for types that support it otherwise @c bad
 *
 * @relates lsst::afw::image::Image
 */
template<typename ImageT>
typename ImageT::SinglePixel badPixel(typename ImageT::Pixel bad=0 ///< The bad value if NaN isn't supported
                                     ) {
    typedef typename ImageT::SinglePixel SinglePixelT;
    return SinglePixelT(std::numeric_limits<SinglePixelT>::has_quiet_NaN ?
                        std::numeric_limits<SinglePixelT>::quiet_NaN() : bad);
}
            
}}}

#endif
