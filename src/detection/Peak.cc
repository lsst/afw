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
 
/*****************************************************************************/
/**
 * \file
 *
 * \brief Handle Peak%s
 */
#include <cassert>
#include <string>
#include <typeinfo>
#include <boost/format.hpp>
#include "lsst/pex/logging/Trace.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/Peak.h"

namespace detection = lsst::afw::detection;

int detection::Peak::id = 0;            //!< Counter for Peak IDs

/**
 * Return a string-representation of a Peak
 */
std::string detection::Peak::toString() {
    return (boost::format("%d: (%d,%d)  (%.3f, %.3f)") % _id % _ix % _iy % _fx % _fy).str();
}
