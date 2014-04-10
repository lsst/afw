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

namespace lsst { namespace afw {

namespace geom {

template <typename T, int N> class Point;

namespace ellipses {

class Quadrupole;

} // namespace ellipses
} // namespace geom

namespace table {

template <typename T>
class PointKey : public FunctorKey< geom::Point<T,2> > {
public:

    PointKey() : _x(), _y() {}

    PointKey(Key<T> const & x, Key<T> const & y) : _x(x), _y(y) {}

    virtual geom::Point<T,2> get(BaseRecord const & record) const;

    virtual void set(BaseRecord & record, geom::Point<T,2> const & value) const;

    Key<T> getX() const { return _x; }
    Key<T> getY() const { return _y; }

private:
    Key<T> _x;
    Key<T> _y;
};

typedef PointKey<int> Point2IKey;
typedef PointKey<double> Point2DKey;

class QuadrupoleKey : public FunctorKey<geom::ellipses::Quadrupole> {
public:

    QuadrupoleKey() : _ixx(), _iyy(), _ixy() {}

    QuadrupoleKey(Key<double> const & ixx, Key<double> const & iyy, Key<double> const & ixy) :
        _ixx(ixx), _iyy(iyy), _ixy(ixy)
    {}

    virtual geom::ellipses::Quadrupole get(BaseRecord const & record) const;

    virtual void set(BaseRecord & record, geom::ellipses::Quadrupole const & value) const;

    Key<double> getIxx() const { return _ixx; }
    Key<double> getIyy() const { return _iyy; }
    Key<double> getIxy() const { return _ixy; }

private:
    Key<double> _ixx;
    Key<double> _iyy;
    Key<double> _ixy;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_aggregates_h_INCLUDED
