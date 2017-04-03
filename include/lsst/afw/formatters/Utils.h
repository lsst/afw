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
//! \file
//! \brief Formatting utilities
//!
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS_UTILS_H
#define LSST_AFW_FORMATTERS_UTILS_H

#include <string>

#include "lsst/base.h"

namespace lsst {
namespace daf {
    namespace base {
        class PropertySet;
    }
    namespace persistence {
        class LogicalLocation;
    }
}
namespace pex {
    namespace policy {
        class Policy;
    }
}
namespace afw {
namespace formatters {

bool extractOptionalFlag(
    CONST_PTR(lsst::daf::base::PropertySet) const& properties,
    std::string const & name
);

std::string const getItemName(
    CONST_PTR(lsst::daf::base::PropertySet) const& properties
);

std::string const getTableName(
    CONST_PTR(lsst::pex::policy::Policy) const& policy,
    CONST_PTR(lsst::daf::base::PropertySet) const& properties
);

std::vector<std::string> getAllSliceTableNames(
    CONST_PTR(lsst::pex::policy::Policy) const& policy,
    CONST_PTR(lsst::daf::base::PropertySet) const& properties
);

void createTable(
    lsst::daf::persistence::LogicalLocation const & location,
    CONST_PTR(lsst::pex::policy::Policy) const& policy,
    CONST_PTR(lsst::daf::base::PropertySet) const& properties
);

void dropAllSliceTables(
    lsst::daf::persistence::LogicalLocation const & location,
    CONST_PTR(lsst::pex::policy::Policy) const & policy,
    CONST_PTR(lsst::daf::base::PropertySet) const & properties
);

int extractSliceId(CONST_PTR(lsst::daf::base::PropertySet) const& properties);
int64_t extractFpaExposureId(CONST_PTR(lsst::daf::base::PropertySet) const& properties);
int64_t extractCcdExposureId(CONST_PTR(lsst::daf::base::PropertySet) const& properties);
int64_t extractAmpExposureId(CONST_PTR(lsst::daf::base::PropertySet) const& properties);
int extractVisitId(CONST_PTR(lsst::daf::base::PropertySet) const& properties);
int extractCcdId(CONST_PTR(lsst::daf::base::PropertySet) const& properties);
int extractAmpId(CONST_PTR(lsst::daf::base::PropertySet) const& properties);

std::string formatFitsProperties(lsst::daf::base::PropertySet const& prop);
int countFitsHeaderCards(lsst::daf::base::PropertySet const& prop);

}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_UTILS_H

