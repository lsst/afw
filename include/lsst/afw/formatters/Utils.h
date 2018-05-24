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
namespace pex {
namespace policy {
class Policy;
}
}  // namespace pex
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

/**
 * Returns the name of the table that a single slice of a pipeline involved in the processing
 * of a single visit should use for persistence of a particular output. All slices can be
 * configured to use the same (per-visit) table name using policy parameters.
 *
 * @param[in] policy   The `Policy` containing the table name pattern ("${itemName}.tableNamePattern",
 *                     where ${itemName} is looked up in `properties` using the "itemName" key)
 *                     from which the the actual table name is derived. This pattern may contain
 * a set of parameters in `%(key)` format - these are interpolated by looking up `"key"` in
 * the `properties` PropertySet.
 *
 * @param[in] properties   Provides runtime specific properties necessary to construct the
 *                         output table name.
 * @returns table name
 */
std::string const getTableName(std::shared_ptr<lsst::pex::policy::Policy const> const& policy,
                               std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);

/**
 * Stores the name of the table that each slice of a pipeline involved in processing a visit
 * used for persistence of its outputs. If slices were configured to all use the same (per-visit)
 * table name, a single name is stored.
 *
 * @param[in] policy   The `Policy` containing the table name pattern ("${itemName}.tableNamePattern",
 *                     where ${itemName} is looked up in `properties` using the "itemName" key)
 *                     from which the the actual table name is derived. This pattern may contain
 * a set of parameters in `%(key)` format - these are interpolated by looking up `"key"` in
 * the `properties` PropertySet.
 *
 * @param[in] properties   The runtime specific properties necessary to construct the table names.
 *
 * string. The `"visitId0"` property must also be present, and shall be a non-negative integer of type
 * `int64_t` uniquely identifying the current LSST visit. If the `"${itemName}.isPerSliceTable"`
 * property is present, is of type `bool` and is set to `true`, then it is assumed that
 * `"${itemName}.numSlices"` (a positive integer of type `int`) output tables exist and
 * are to be read in.
 *
 * @returns a list of table names
 * @see getTableName()
 */
std::vector<std::string> getAllSliceTableNames(
        std::shared_ptr<lsst::pex::policy::Policy const> const& policy,
        std::shared_ptr<lsst::daf::base::PropertySet const> const& properties);

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
