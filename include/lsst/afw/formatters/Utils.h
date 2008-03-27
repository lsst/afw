// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//! \file   Utils.h
//##====----------------                                ----------------====##/

#ifndef LSST_FW_FORMATTERS_UTILS_H
#define LSST_FW_FORMATTERS_UTILS_H

#include <string>
#include <lsst/daf/data/DataProperty.h>
#include <lsst/pex/policy/Policy.h>
#include <lsst/pex/persistence/LogicalLocation.h>


namespace lsst {
namespace fw {
namespace formatters {

using namespace lsst::pex::persistence;
using lsst::daf::data::DataProperty;
using lsst::pex::policy::Policy;

bool extractOptionalFlag(
    DataProperty::PtrType const & properties,
    std::string           const & name
);

std::string const extractPolicyString(
    Policy::Ptr const & policy,
    std::string const & key,
    std::string const & def
);

std::string const getItemName(DataProperty::PtrType const & properties);

std::string const getVisitSliceTableName(
    Policy::Ptr           const & policy,
    DataProperty::PtrType const & properties
);

void getAllVisitSliceTableNames(
    std::vector<std::string>    & names,
    Policy::Ptr           const & policy,
    DataProperty::PtrType const & properties
);

void createVisitSliceTable(
    LogicalLocation       const & location,
    Policy::Ptr           const & policy,
    DataProperty::PtrType const & properties
);

void dropAllVisitSliceTables(
    LogicalLocation       const & location,
    Policy::Ptr           const & policy,
    DataProperty::PtrType const & properties
);

int extractSliceId(DataProperty::PtrType const& properties);
int64_t extractExposureId(DataProperty::PtrType const& properties);
int extractVisitId(DataProperty::PtrType const& properties);
int extractCcdId(DataProperty::PtrType const& properties);
int64_t extractCcdExposureId(DataProperty::PtrType const& properties);

}}} // end of namespace lsst::afw::formatters

#endif // LSST_FW_FORMATTERS_UTILS_H

