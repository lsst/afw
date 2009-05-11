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
