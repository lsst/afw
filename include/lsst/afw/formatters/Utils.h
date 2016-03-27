// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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

std::string formatFitsProperties(CONST_PTR(lsst::daf::base::PropertySet) const& prop);
int countFitsHeaderCards(CONST_PTR(lsst::daf::base::PropertySet) const& prop);

}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_UTILS_H

