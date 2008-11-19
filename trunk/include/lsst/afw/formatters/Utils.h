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

#include "lsst/daf/base/DataProperty.h"
#include "lsst/daf/persistence/LogicalLocation.h"
#include "lsst/pex/policy/Policy.h"


namespace lsst {
namespace afw {
namespace formatters {

bool extractOptionalFlag(
    lsst::daf::base::DataProperty::PtrType const & properties,
    std::string const & name
);

std::string const extractPolicyString(
    lsst::pex::policy::Policy::Ptr const & policy,
    std::string const & key,
    std::string const & def
);

std::string const getItemName(
    lsst::daf::base::DataProperty::PtrType const & properties
);

std::string const getVisitSliceTableName(
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::DataProperty::PtrType const & properties
);

void getAllVisitSliceTableNames(
    std::vector<std::string> & names,
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::DataProperty::PtrType const & properties
);

void createVisitSliceTable(
    lsst::daf::persistence::LogicalLocation const & location,
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::DataProperty::PtrType const & properties
);

void dropAllVisitSliceTables(
    lsst::daf::persistence::LogicalLocation const & location,
    lsst::pex::policy::Policy::Ptr const & policy,
    lsst::daf::base::DataProperty::PtrType const & properties
);

int extractSliceId(lsst::daf::base::DataProperty::PtrType const& properties);
int64_t extractExposureId(lsst::daf::base::DataProperty::PtrType const& properties);
int extractVisitId(lsst::daf::base::DataProperty::PtrType const& properties);
int extractCcdId(lsst::daf::base::DataProperty::PtrType const& properties);
int64_t extractCcdExposureId(lsst::daf::base::DataProperty::PtrType const& properties);

}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_UTILS_H

