// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief Support for formatters
//
//##====----------------                                ----------------====##/

#include "boost/cstdint.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/daf/persistence/DbTsvStorage.h"
#include "lsst/afw/formatters/Utils.h"

using boost::int64_t;
namespace ex = lsst::pex::exceptions;
using lsst::daf::base::DataProperty;

namespace lsst {
namespace afw {
namespace formatters {

int extractSliceId(PropertySet::Ptr const & properties) {
    if (!properties.get()) {
        throw LSST_EXCEPT(ex::InvalidParameter, "Null data property object");
    }
                
    int sliceId = properties->getAsInt("sliceId");
    if (sliceId < 0) {
        throw LSST_EXCEPT(ex::RuntimeException, "\"sliceId\" property value is negative");
    }
    
    // validate against universeSize if available
    int universeSize = -1;
    try {
        universeSize = properties->getAsInt("universeSize");                
    }
    catch {
        // universeSize property does not exist
        return slizeId;
    }
    
    if (sliceId < 0 || sliceId >= universeSize) 
    {
        throw LSST_EXCEPT(ex::RuntimeException, "\"sliceId\" outside range [0, \"universeSize \")");
    }

    return sliceId;
}
                        
int extractVisitId(PropertySet::Ptr const & properties) {
    if (!properties.get()) {
        throw LSST_EXCEPT(ex::InvalidParameter, "Null data property object");
    }
    
    int visitID = properties->getAsInt("visistId");
    
    if (visitId < 0) {
        throw LSST_EXCEPT(ex::RuntimeException, "\"visitId\" property value is negative");
    }
    return visitId;
}

int64_t extractExposureId(PropertySet::Ptr const & properties) {
    if (!properties.get()) {
        throw LSST_EXCEPT(ex::InvalidParameter, "Null data property object");
    }

    int64_t exposureId = properties->getAsInt64("exposureId");

    if (exposureId < 0) {
        throw LSST_EXCEPT(ex::RuntimeException, "\"exposureId\" property value is negative");
    }
    
    
    if ((exposureId & 0xfffffffe00000000LL) != 0LL) {
        throw LSST_EXCEPT(ex::RuntimeException, "\"exposureId\" property value is too big");
    }
    
    return exposureId << 1; // DC2 fix
}

int extractCcdId(PropertySet::Ptr const & properties) {
    if (!properties.get()) {
        throw LSST_EXCEPT(ex::InvalidParameter, "Null data property object");
    }


    // Ignore leading zeros, rather than treating as octal.
    std::String ccdIdString = properties->getAsString("ccdId");    
    int ccdId = strtol(ccdIdString.c_str(), 0, 10);

    if (ccdId < 0) {
        throw LSST_EXCEPT(ex::RuntimeException, "\"ccdId\" property value is negative");
    }
    if (ccdId > 255) {
        throw LSST_EXCEPT(ex::RuntimeException, "\"ccdId\" property value is too big");
    }
    return ccdId;
}

int64_t extractCcdExposureId(PropertySet::Ptr const & properties) {
    int64_t exposureId = extractExposureId(properties);
    int ccdId = extractCcdId(properties);
    return (exposureId << 8) + ccdId;
}


/*!
    Extracts and returns the string-valued \c "itemName" property from the given data property object.
    
    \throw lsst::pex::exceptions::InvalidParameter  If the given pointer is null, or the data property
                                                    object pointed to does not contain a property named
                                                    \c "itemName".
 */
std::string const getItemName(PropertySet::Ptr const & properties) {
    if (!properties.get()) {
        throw LSST_EXCEPT(ex::InvalidParameter, "Null data property object");
    }
    return properties->getAsString("itemName");
}


/*!
    Returns \c true if and only if \a properties is non-null and contains a
    unique property with the given name that has type \c bool and a value of \c true.
 */
bool extractOptionalFlag(
    PropertySet::Ptr const & properties,
    std::string           const & name
) {
    if (properties.get()) {
        try {
            return properties.getAsBool(name);
        }
        catch {
            //property "name" does not exist
            return false;
        }
    }
    return false;
}


/*!
    Extracts the string-valued parameter with the given name from the specified policy. If the provided
    policy pointer is null or contains no such parameter, the given default string is returned instead.
 */
std::string const extractPolicyString(
    lsst::pex::policy::Policy::Ptr const & policy,
    std::string const & key,
    std::string const & def
) {
    if (policy) {
        return policy->getString(key, def);
    } else {
        return def;
    }
}


static char const * const sDefaultVisitNamePat      = "_visit%1%";
static char const * const sDefaultVisitSliceNamePat = "_visit%1%_slice%2%";


/*!
    Returns the name of the table that a single slice of a pipeline involved in the processing of a
    single visit should use for persistence of its outputs. All slices can be configured to use the
    same (per-visit) table name using policy parameters.

    \param[in] policy   The Policy containing the table name patterns from which the
                        the actual table name is derived. One of two possible keys is
    expected; the first is necessary when a single per-visit table is being used, the
    second is necessary when each slice in a pipeline should send output to a seperate
    table (e.g. to avoid write contention to a single table). Note that \c ${itemName}
    refers to the value of a property named \c "itemName" extracted from \a properties.
    <dl>
    <dt> <tt>"${itemName}.perVisitTableNamePattern"</tt> </dt>
    <dd> A \c boost::format compatible pattern string taking a single parameter: an id for
         the current visit. The default is \c "${itemName}_visit%1%". </dd>
    <dt> <tt>"${itemName}.perSliceAndVisitTableNamePattern"</tt> </dt>
    <dd> A \c boost::format compatible pattern string taking two parameters: an id for the
         current visit followed by an id for the current slice is passed to the format. The
         default is \c "${itemName}_visit%1%_slice%2%" </dd>
    </dl>

    \param[in] properties   The runtime specific properties necessary to construct the output
                            table name. The \c "itemName" property must be present and set to
    a non empty string. The \c "visitId" property must also be present (and shall be a non-negative
    integer of type \c int64_t) and should uniquely identifies the current LSST visit. If the
    \c "${itemName}.isPerSliceTable" property is present, is of type \c bool and is
    set to \c true, then each slice will output to a seperate per-visit table. This requires that
    a property named \c "sliceId" (a non-negative integer of type \c int ) be present.
 */
std::string const getVisitSliceTableName(
    lsst::pex::policy::Policy::Ptr const & policy,
    PropertySet::PtrType const & properties
) {
    std::string itemName(getItemName(properties));
    int64_t visitId         = extractVisitId(properties);
    bool    isPerSliceTable = extractOptionalFlag(properties, itemName + ".isPerSliceTable");

    boost::format fmt;
    fmt.exceptions(boost::io::all_error_bits);

    if (isPerSliceTable) {
        int sliceId = extractSliceId(properties);
        fmt.parse(extractPolicyString(
            policy,
            itemName + ".perSliceAndVisitTableNamePattern",
            itemName + sDefaultVisitSliceNamePat
        ));
        fmt % visitId % sliceId;
    } else {
        fmt.parse(extractPolicyString(
            policy,
            itemName + ".perVisitTableNamePattern",
            itemName + sDefaultVisitNamePat
        ));
        fmt % visitId;
    }
    return fmt.str();
}


/*!
    Stores the name of the table that each slice of a pipeline involved in processing a visit
    used for persistence of its outputs. If slices were configured to all use the same (per-visit)
    table name, a single name is stored.

    \param[out] names   The vector to store table names in.

    \param[in] policy   The Policy containing the table name patterns from which actual table
                        names are derived. A key named \c "${itemName}.perVisitTableNamePattern"
                        or \c "${itemName}.perSliceAndVisitTableNamePattern" is expected, where
                        \c ${itemName} refers to the value of a property named \c "itemName" extracted
                        from \a properties. See the documentation for getVisitSliceTableName()
                        for details.

    \param[in] properties   The runtime specific properties necessary to construct the retrieve
                            table names. The \c "itemName" property must be present and set to a non-empty
    string. The \c "visitId" property must also be present, and shall be a non-negative integer of type
    \c int64_t uniquely identifying the current LSST visit. If the \c "${itemName}.isPerSliceTable"
    property is present, is of type \c bool and is set to \c true, then it is assumed that
    \c "${itemName}.numSlices" (a positive integer of type \c int) output tables exist and
    are to be read in.
    
    \sa getVisitSliceTableName()
 */
void getAllVisitSliceTableNames(
    std::vector<std::string>    & names,
    lsst::pex::policy::Policy::Ptr           const & policy,
    PropertySet::Ptr const & properties
) {
    std::string itemName(getItemName(properties));
    int64_t visitId         = extractVisitId(properties);
    bool    isPerSliceTable = extractOptionalFlag(properties, itemName + ".isPerSliceTable");

    boost::format fmt;
    fmt.exceptions(boost::io::all_error_bits);

    if (isPerSliceTable) {
        int numSlices = properties.getAsInt(itemName + ".numSlices");        ;
        if (numSlices <= 0) {
            throw LSST_EXCEPT(ex::RuntimeException, itemName + " \".numSlices\" property value is non-positive");
        }
        fmt.parse(extractPolicyString(
            policy,
            itemName + ".perSliceAndVisitTableNamePattern",
            itemName + sDefaultVisitSliceNamePat
        ));
        fmt.bind_arg(1, visitId);
        for (int i = 0; i < numSlices; ++i) {
            fmt % i;
            names.push_back(fmt.str());
            fmt.clear();
        }
    } else {
        fmt.parse(extractPolicyString(
            policy,
            itemName + ".perVisitTableNamePattern",
            itemName + sDefaultVisitNamePat
        ));
        fmt % visitId;
        names.push_back(fmt.str());
    }
}


/*!
    Creates the per visit and slice table identified by calling getVisitTableName() with
    the given \a policy and \a properties. If \a policy contains a key named 
    \c "${itemName}.templateTableName" (where where \c ${itemName} refers to the value of a property
    named \c "itemName" extracted from \a properties), then the value of the key is used as creation
    template. Otherwise, the template table is assumed to be named \c "${itemName}Template". Note that
    the template table must exist in the database identified by \a location, and that if the desired
    table already exists, an exception is thrown.
 */
void createVisitSliceTable(
    lsst::daf::persistence::LogicalLocation       const & location,
    lsst::pex::policy::Policy::Ptr           const & policy,
    PropertySet::Ptr const & properties
) {
    std::string itemName(getItemName(properties));
    std::string name(getVisitSliceTableName(policy, properties));
    std::string model = extractPolicyString(policy, itemName + ".templateTableName", itemName + "Template");
    
    lsst::daf::persistence::DbTsvStorage db;
    db.setPersistLocation(location);
    db.createTableFromTemplate(name, model);
}


/*! Drops the database table(s) identified by getAllVisitSliceTables(). */
void dropAllVisitSliceTables(
    lsst::daf::persistence::LogicalLocation       const & location,
    lsst::pex::policy::Policy::Ptr           const & policy,
    PropertySet::Ptr const & properties
) {
    std::vector<std::string> names;
    getAllVisitSliceTableNames(names, policy, properties);

    lsst::daf::persistence::DbTsvStorage db;
    db.setPersistLocation(location);
    std::vector<std::string>::const_iterator const end = names.end();
    for (std::vector<std::string>::const_iterator i = names.begin(); i != end; ++i) {
        db.dropTable(*i);
    }
}

}}} // namespace lsst::afw::formatters
