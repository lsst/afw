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
 
#ifndef LSST_AFW_GEOM_Span_h_INCLUDED
#define LSST_AFW_GEOM_Span_h_INCLUDED

#include <string>

#include "boost/serialization/nvp.hpp"

#include "lsst/base.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace boost{
namespace serialization{
    class access;
}}

namespace lsst {
namespace afw { 

namespace detection {
class Footprint;
} // namespace detection

namespace geom {

/*!
 * \brief A range of pixels within one row of an Image
 */
class Span {
public:

    typedef boost::shared_ptr<Span> Ptr;
    typedef boost::shared_ptr<Span const> ConstPtr;

    Span(int y,                         //!< Row that Span's in
         int x0,                        //!< Starting column (inclusive)
         int x1)                        //!< Ending column (inclusive)
        : _y(y), _x0(x0), _x1(x1) {}

    int getX0() const { return _x0; }         ///< Return the starting x-value
    int getX1() const { return _x1; }         ///< Return the ending x-value
    int getY()  const { return _y; }          ///< Return the y-value
    int getWidth() const { return _x1 - _x0 + 1; } ///< Return the number of pixels

    bool contains(int x) { return (x >= _x0) && (x <= _x1); }
    bool contains(int x, int y) { return (x >= _x0) && (x <= _x1) && (y == _y); }

    /// Return a string-representation of a Span
    std::string toString() const;

    void shift(int dx, int dy) { _x0 += dx; _x1 += dx; _y += dy; }


    /* Required to make Span "LessThanComparable" so they can be used
     * in sorting, binary search, etc.
     * http://www.sgi.com/tech/stl/LessThanComparable.html
     */
    bool operator<(const Span& b) const;

    friend class detection::Footprint;
private:
    Span() {}

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & boost::serialization::make_nvp("y", _y)
            & boost::serialization::make_nvp("x0", _x0)
            & boost::serialization::make_nvp("x1", _x1);
    }

    int _y;                             //!< Row that Span's in
    int _x0;                            //!< Starting column (inclusive)
    int _x1;                            //!< Ending column (inclusive)
};

}}} // lsst::afw::geom

#endif // LSST_AFW_GEOM_Span_h_INCLUDED
