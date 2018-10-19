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

//
//##====----------------                                ----------------====##/
//
//        Formatting utilities
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS_UTILS_H
#define LSST_AFW_FORMATTERS_UTILS_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "ndarray.h"

#include "lsst/base.h"
#include "lsst/daf/base.h"

namespace lsst {
namespace daf {
namespace base {
class PropertySet;
}
namespace persistence {
class LogicalLocation;
}
}  // namespace daf
namespace afw {
namespace formatters {

/**
 * Returns `true` if and only if `properties` is non-null and contains a
 * unique property with the given name that has type `bool` and a value of `true`.
 */
bool extractOptionalFlag(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties,
                         std::string const& name);

/**
 * Extracts and returns the string-valued `"itemName"` property from the given data property object.
 *
 * @throws lsst::pex::exceptions::InvalidParameterError
 *        If the given pointer is null, or the `PropertySet` pointed
 *        to does not contain a unique property named `"itemName"`.
 */
std::string const getItemName(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);

int extractSliceId(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);
int64_t extractFpaExposureId(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);
int64_t extractCcdExposureId(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);
int64_t extractAmpExposureId(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);
int extractVisitId(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);
int extractCcdId(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);
int extractAmpId(std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);

int countFitsHeaderCards(lsst::daf::base::PropertySet const& prop);

/**
 * Encode a std::string as a vector of uint8
 */
ndarray::Array<std::uint8_t, 1, 1> stringToBytes(std::string const& str);

/**
 * Decode a std::string from a vector of uint8 returned by stringToBytes
 */
std::string bytesToString(ndarray::Array<std::uint8_t const, 1, 1> const& bytes);

}  // namespace formatters
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_FORMATTERS_UTILS_H
