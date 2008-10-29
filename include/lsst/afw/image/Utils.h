// -*- lsst-c++ -*-

#ifndef LSST_AFW_IMAGE_UTILS_H
#define LSST_AFW_IMAGE_UTILS_H

#include <list>
#include <map>
#include <string>
#include <utility>

#include "boost/mpl/bool.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/afw/image/lsstGil.h"
#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/formatters/ImageFormatter.h"

namespace lsst { namespace afw { namespace image {
    /************************************************************************************************************/
    /// @brief a single Point, templated over the type to be used to store the coordinates
    template<typename T>
    class Point {
    public:
        Point(T val=0) : _x(val), _y(val) {}
        Point(T x, T y) : _x(x), _y(y) {}
        Point(T const xy[2]) : _x(xy[0]), _y(xy[1]) {}

        T getX() const { return _x; }
        T getY() const { return _y; }
        void setX(T x) { _x = x; }
        void setY(T y) { _y = y; }

        bool operator==(const Point& rhs) const { return (_x == rhs._x && _y == rhs._y); }
        bool operator!=(const Point& rhs) const { return !(*this == rhs); }
        
        Point operator+(const Point& rhs) const { return Point(_x + rhs._x, _y + rhs._y); }
        Point operator-(const Point& rhs) const { return Point(_x - rhs._x, _y - rhs._y); }
#if !defined(SWIG)
        T const& operator[](int const i) const {
            switch (i) {
              case 0: return _x;
              case 1: return _y;
              default: throw lsst::pex::exceptions::OutOfRange(boost::format("Index i == %d must be 0 or 1") % i);
            }
        }
        T& operator[](int const i) {
            return const_cast<T&>((static_cast<const Point&>(*this))[i]); // Meyers, Effective C++, Item 3
        }
#endif
    private:
        T _x, _y;
    };
    typedef Point<double> PointD;
    typedef Point<int> PointI;
    /// @brief A bounding box
    class BBox : private std::pair<PointI, PointI > {
    public:
        //! Create a BBox with origin llc and the specified dimensions
        BBox(PointI llc=PointI(0,0),       ///< Desired lower left corner
             int width=0,                  ///< Width of BBox (pixels)
             int height=0                  ///< Height of BBox (pixels)
            ) :
            std::pair<PointI, PointI>(llc, PointI(width, height)) {} 
        //! Create a BBox given two corners
        BBox(PointI llc,                ///< Desired lower left corner
             PointI urc                 ///< Desired upper right corner
            ) :
            std::pair<PointI, PointI>(llc, urc - llc + 1) {}

        //! Return true iff the point lies in the BBox
        bool contains(PointI p) {           ///< The point to check
            return p.getX() >= getX0() && p.getX() <= getX1() && p.getY() >= getY0() && p.getY() <= getY1();
        }

        //! Grow the BBox to include the specified PointI
        void grow(PointI p) {           ///< The point to include
            if (getWidth() == 0 && getHeight() == 0) {
                first.setX(p.getX());
                first.setY(p.getY());
                second.setX(1);
                second.setY(1);

                return;
            }

            if (p.getX() < getX0()) {
                second.setX(getWidth() + (getX0() - p.getX()));
                first.setX(p.getX());
            } else if (p.getX() > getX1()) {
                second.setX(p.getX() - getX0() + 1);
            }

            if (p.getY() < getY0()) {
                second.setY(getHeight() + (getY0() - p.getY()));
                first.setY(p.getY());
            } else if (p.getY() > getY1()) {
                second.setY(p.getY() - getY0() + 1);
            }
        }
        //! Offset a BBox by the specified vector
        void shift(int dx,               ///< How much to offset in x
                   int dy                ///< How much to offset in y
                  ) {
            first.setX(first.getX() + dx);
            first.setY(first.getY() + dy);
            second.setX(second.getX());
            second.setY(second.getY());

            }

        int getX0() const { return first.getX(); }
        int getY0() const { return first.getY(); }
        int getX1() const { return first.getX() + second.getX() - 1; }
        int getY1() const { return first.getY() + second.getY() - 1; }
        int getWidth() const { return second.getX(); }
        int getHeight() const { return second.getY(); }
        const std::pair<int, int> dimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }

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
     * Only integer centers are supported.  It'd be easy enough to add other
     * types, but as BCircles are designed by analogy to BBoxes to define
     * sets of pixels, I haven't done so
     */
    class BCircle : private std::pair<PointI, float > {
    public:
        /// Create a BCircle given centre and radius
        BCircle(PointI center,               //!< Centre of circle
                float r)                     //!< Radius of circle
            : std::pair<PointI, float>(center, r) {}

        /// Return the circle's centre
        PointI const& getCenter() const { return first; }
        /// Return the circle's radius
        float getRadius() const { return second; }
    };
}}}

#endif
