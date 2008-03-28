// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//! \file   Utils.h
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS_UTILS_H
#define LSST_AFW_FORMATTERS_UTILS_H

#include <string>
#include <lsst/daf/data/DataProperty.h>
#include <lsst/daf/policy/Policy.h>
#include <lsst/daf/persistence/LogicalLocation.h>


namespace lsst {
namespace afw {
namespace formatters {

using namespace lsst::daf::persitence;
using lsst::daf::data::DataProperty;
using lsst::daf::policy::Policy;

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

}}} // lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_UTILS_H

