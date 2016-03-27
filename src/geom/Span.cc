// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "boost/format.hpp"

#include "lsst/afw/geom/Span.h"

namespace lsst { namespace afw { namespace geom {

bool Span::operator<(const Span& b) const {
	if (_y < b._y)
		return true;
	if (_y > b._y)
		return false;
	// y equal; check x0...
	if (_x0 < b._x0)
		return true;
	if (_x0 > b._x0)
		return false;
	// x0 equal; check x1...
	if (_x1 < b._x1)
		return true;
	return false;
}

std::string Span::toString() const {
    return str(boost::format("%d: %d..%d") % _y % _x0 % _x1);
}

}}} // namespace lsst::afw::geom
