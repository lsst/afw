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

#include "boost/format.hpp"

#include "lsst/afw/geom/Span.h"

namespace lsst {
namespace afw {
namespace geom {

bool Span::operator<(const Span& b) const {
    if (_y < b._y) return true;
    if (_y > b._y) return false;
    // y equal; check x0...
    if (_x0 < b._x0) return true;
    if (_x0 > b._x0) return false;
    // x0 equal; check x1...
    if (_x1 < b._x1) return true;
    return false;
}

std::string Span::toString() const { return str(boost::format("%d: %d..%d") % _y % _x0 % _x1); }
}
}
}  // namespace lsst::afw::geom
