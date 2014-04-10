// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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
#ifndef AFW_TABLE_aggregates_h_INCLUDED
#define AFW_TABLE_aggregates_h_INCLUDED

#include "lsst/afw/table/FunctorKey.h"
#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw {

namespace geom {

template <typename T, int N> class Point;

namespace ellipses {

class Quadrupole;

} // namespace ellipses
} // namespace geom

namespace table {

/**
 *  @brief A FunctorKey used to get or set a geom::Point from an (x,y) pair of int or double Keys.
 */
template <typename T>
class PointKey : public FunctorKey< lsst::afw::geom::Point<T,2> > {
public:

    /// Default constructor; instance will not be usuable unless subsequently assigned to.
    PointKey() : _x(), _y() {}

    /// Construct from a pair of Keys
    PointKey(Key<T> const & x, Key<T> const & y) : _x(x), _y(y) {}

    /**
     *  @brief Construct from a subschema, assuming .x and .y subfields
     *
     *  If a schema has "a.x" and "a.y" fields, this constructor allows you to construct
     *  a PointKey via:
     *  @code
     *  PointKey<T> k(schema["a"]);
     *  @endcode
     */
    PointKey(SubSchema const & s) : _x(s["x"]), _y(s["y"]) {}

    /// Get a Point from the given record
    virtual geom::Point<T,2> get(BaseRecord const & record) const;

    /// Set a Point in the given record
    virtual void set(BaseRecord & record, geom::Point<T,2> const & value) const;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying x and y Keys
    bool operator==(PointKey<T> const & other) const { return _x == other._x && _y == other._y; }
    bool operator!=(PointKey<T> const & other) const { return !(*this == other); }
    //@}

    /// Return True if both the x and y Keys are valid.
    bool isValid() const { return _x.isValid() && _y.isValid(); }

    /// Return the underlying x Key
    Key<T> getX() const { return _x; }

    /// Return the underlying y Key
    Key<T> getY() const { return _y; }

private:
    Key<T> _x;
    Key<T> _y;
};

typedef PointKey<int> Point2IKey;
typedef PointKey<double> Point2DKey;

/**
 *  @brief A FunctorKey used to get or set a geom::ellipses::Quadrupole from an (xx,yy,xy) tuple of Keys.
 */
class QuadrupoleKey : public FunctorKey< lsst::afw::geom::ellipses::Quadrupole > {
public:

    /// Default constructor; instance will not be usuable unless subsequently assigned to.
    QuadrupoleKey() : _ixx(), _iyy(), _ixy() {}

    /// Construct from individual Keys
    QuadrupoleKey(Key<double> const & ixx, Key<double> const & iyy, Key<double> const & ixy) :
        _ixx(ixx), _iyy(iyy), _ixy(ixy)
    {}

    /**
     *  @brief Construct from a subschema, assuming .xx, .yy, and .xy subfields
     *
     *  If a schema has "a.xx", "a.yy", and "a.xy" fields, this constructor allows you to construct
     *  a QuadrupoleKey via:
     *  @code
     *  QuadrupoleKey k(schema["a"]);
     *  @endcode
     */
    QuadrupoleKey(SubSchema const & s) : _ixx(s["xx"]), _iyy(s["yy"]), _ixy(s["xy"]) {}

    /// Get a Quadrupole from the given record
    virtual geom::ellipses::Quadrupole get(BaseRecord const & record) const;

    /// Set a Quadrupole in the given record
    virtual void set(BaseRecord & record, geom::ellipses::Quadrupole const & value) const;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying Ixx, Iyy, Ixy Keys
    bool operator==(QuadrupoleKey const & other) const {
        return _ixx == other._ixx && _iyy == other._iyy && _ixy == other._ixy;
    }
    bool operator!=(QuadrupoleKey const & other) const { return !(*this == other); }
    //@}

    /// Return True if all the constituent Keys are valid.
    bool isValid() const { return _ixx.isValid() && _iyy.isValid() && _ixy.isValid(); }

    //@{
    /// Return a constituent Key
    Key<double> getIxx() const { return _ixx; }
    Key<double> getIyy() const { return _iyy; }
    Key<double> getIxy() const { return _ixy; }
    //@}

private:
    Key<double> _ixx;
    Key<double> _iyy;
    Key<double> _ixy;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_aggregates_h_INCLUDED
