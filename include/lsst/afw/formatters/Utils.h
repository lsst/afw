// -*- lsst-c++ -*-
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

std::string const extractPolicyString(
    lsst::pex::policy::Policy::Ptr const & policy,
    std::string const & key,
    std::string const & def
);

std::string const getItemName(
    lsst::daf::base::PropertySet::Ptr const & properties
);

std::string const getVisitSliceTableName(
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::PropertySet::Ptr const & properties
);

void getAllVisitSliceTableNames(
    std::vector<std::string> & names,
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::PropertySet::Ptr const & properties
);

void createVisitSliceTable(
    lsst::daf::persistence::LogicalLocation const & location,
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::PropertySet::Ptr const & properties
);

void dropAllVisitSliceTables(
    lsst::daf::persistence::LogicalLocation const & location,
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::PropertySet::Ptr const & properties
);

int extractSliceId(lsst::daf::base::PropertySet::Ptr const& properties);
int64_t extractExposureId(lsst::daf::base::PropertySet::Ptr const& properties);
int64_t extractCcdExposureId(lsst::daf::base::PropertySet::Ptr const& properties);
int64_t extractAmpExposureId(lsst::daf::base::PropertySet::Ptr const& properties);
int extractVisitId(lsst::daf::base::PropertySet::Ptr const& properties);
int extractCcdId(lsst::daf::base::PropertySet::Ptr const& properties);
int extractAmpId(lsst::daf::base::PropertySet::Ptr const& properties);

std::string const formatFitsProperties(lsst::daf::base::PropertySet::Ptr prop);
int countFitsHeaderCards(lsst::daf::base::PropertySet::Ptr prop);

}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_UTILS_H

