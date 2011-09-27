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

#include "lsst/daf/base/PropertySet.h"
#include "lsst/daf/persistence/LogicalLocation.h"
#include "lsst/pex/policy/Policy.h"


namespace lsst {
namespace afw {
namespace formatters {

bool extractOptionalFlag(
    lsst::daf::base::PropertySet::Ptr const & properties,
    std::string const & name
);

std::string const getItemName(
    lsst::daf::base::PropertySet::Ptr const & properties
);

std::string const getTableName(
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::PropertySet::Ptr const & properties
);

std::vector<std::string> getAllSliceTableNames(
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::PropertySet::Ptr const & properties
);

void createTable(
    lsst::daf::persistence::LogicalLocation const & location,
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::PropertySet::Ptr const & properties
);

void dropAllSliceTables(
    lsst::daf::persistence::LogicalLocation const & location,
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::PropertySet::Ptr const & properties
);

int extractSliceId(lsst::daf::base::PropertySet::Ptr const& properties);
int64_t extractFpaExposureId(lsst::daf::base::PropertySet::Ptr const& properties);
int64_t extractCcdExposureId(lsst::daf::base::PropertySet::Ptr const& properties);
int64_t extractAmpExposureId(lsst::daf::base::PropertySet::Ptr const& properties);
int extractVisitId(lsst::daf::base::PropertySet::Ptr const& properties);
int extractCcdId(lsst::daf::base::PropertySet::Ptr const& properties);
int extractAmpId(lsst::daf::base::PropertySet::Ptr const& properties);

std::string const formatFitsProperties(lsst::daf::base::PropertySet::Ptr prop);
int countFitsHeaderCards(lsst::daf::base::PropertySet::Ptr prop);

}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_UTILS_H

