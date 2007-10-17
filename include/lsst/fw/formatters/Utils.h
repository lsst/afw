// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//! \file   Utils.h
//##====----------------                                ----------------====##/

#ifndef LSST_FW_FORMATTERS_UTILS_H
#define LSST_FW_FORMATTERS_UTILS_H

#include <string>
#include <lsst/mwi/data/DataProperty.h>
#include <lsst/mwi/policy/Policy.h>
#include <lsst/mwi/persistence/LogicalLocation.h>


namespace lsst {
namespace fw {
namespace formatters {

using namespace lsst::mwi::persistence;
using lsst::mwi::data::DataProperty;
using lsst::mwi::policy::Policy;

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


}}} // end of namespace lsst::fw::formatters

#endif // LSST_FW_FORMATTERS_UTILS_H

