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

#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw { namespace table {

template <typename T>
geom::Point<T,2> PointKey<T>::get(BaseRecord const & record) const {
    return geom::Point<T,2>(record.get(_x), record.get(_y));
}

template <typename T>
void PointKey<T>::set(BaseRecord & record, geom::Point<T,2> const & value) const {
    record.set(_x, value.getX());
    record.set(_y, value.getY());
}

geom::ellipses::Quadrupole QuadrupoleKey::get(BaseRecord const & record) const {
    return geom::ellipses::Quadrupole(record.get(_ixx), record.get(_iyy), record.get(_ixy));
}

void QuadrupoleKey::set(BaseRecord & record, geom::ellipses::Quadrupole const & value) const {
    record.set(_ixx, value.getIxx());
    record.set(_iyy, value.getIyy());
    record.set(_ixy, value.getIxy());
}

template class PointKey<int>;
template class PointKey<double>;

}}} // namespace lsst::afw::table
