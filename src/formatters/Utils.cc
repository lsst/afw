// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief Support for formatters
//
//##====----------------                                ----------------====##/

#include "boost/cstdint.hpp"
#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/daf/persistence/DbTsvStorage.h"
#include "lsst/afw/formatters/Utils.h"

using boost::int64_t;
namespace ex = lsst::pex::exceptions;
using lsst::daf::base::PropertySet;
using lsst::pex::policy::Policy;

namespace lsst {
namespace afw {
namespace formatters {

int extractSliceId(PropertySet::Ptr const & properties) {
    if (properties->isArray("sliceId")) {
        throw LSST_EXCEPT(ex::RuntimeErrorException, "\"sliceId\" property has multiple values");
    }
    int sliceId = properties->getAsInt("sliceId");
    if (sliceId < 0) {
        throw LSST_EXCEPT(ex::RangeErrorException, "negative \"sliceId\"");
    }
    if (properties->exists("universeSize") && !properties->isArray("universeSize")) {
        int universeSize = properties->getAsInt("universeSize");
        if (sliceId >= universeSize) {
            throw LSST_EXCEPT(ex::RangeErrorException, "\"sliceId\" must be less than \"universeSize \"");
        }
    }
    return sliceId;
}
                        
int extractVisitId(PropertySet::Ptr const & properties) {
    if (properties->isArray("visitId")) {
        throw LSST_EXCEPT(ex::RuntimeErrorException, "\"visitId\" property has multiple values");
    }
    int visitId = properties->getAsInt("visitId");
    if (visitId < 0) {
        throw LSST_EXCEPT(ex::RangeErrorException, "negative \"visitId\"");
    }
    return visitId;
}

int64_t extractExposureId(PropertySet::Ptr const & properties) {
    if (properties->isArray("exposureId")) {
        throw LSST_EXCEPT(ex::RuntimeErrorException, "\"exposureId\" property has multiple values");
    }
    int64_t exposureId = properties->getAsInt64("exposureId");
    if (exposureId < 0) {
        throw LSST_EXCEPT(ex::RangeErrorException, "negative \"exposureId\"");
    }
    if ((exposureId & 0xfffffffe00000000LL) != 0LL) {
        throw LSST_EXCEPT(ex::RangeErrorException, "\"exposureId\" is too large");
    }
    return exposureId << 1; // DC2 fix
}

int extractCcdId(PropertySet::Ptr const & properties) {
    if (properties->isArray("ccdId")) {
        throw LSST_EXCEPT(ex::RuntimeErrorException, "\"ccdId\" property has multiple values");
    }
    int ccdId = properties->getAsInt64("ccdId");
    if (ccdId < 0) {
        throw LSST_EXCEPT(ex::RangeErrorException, "negative \"exposureId\"");
    }
    if (ccdId > 255) {
        throw LSST_EXCEPT(ex::RangeErrorException, "\"ccdId\" is too large");
    }
    return ccdId;
}

int64_t extractCcdExposureId(PropertySet::Ptr const & properties) {
    int64_t exposureId = extractExposureId(properties);
    int ccdId = extractCcdId(properties);
    return (exposureId << 8) + ccdId;
}


/**
 * Extracts and returns the string-valued @c "itemName" property from the given data property object.
 *  
 * @throw lsst::pex::exceptions::InvalidParameterException
 *        If the given pointer is null, or the @c PropertySet pointed
 *        to does not contain a unique property named @c "itemName".
 */
std::string const getItemName(PropertySet::Ptr const & properties) {
    if (!properties) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Null PropertySet::Ptr");
    }
    if (properties->isArray("itemName")) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "\"itemName\" property has multiple values");
    } 
    return properties->getAsString("itemName");
}


/**
 * Returns @c true if and only if @a properties is non-null and contains a
 * unique property with the given name that has type @c bool and a value of @c true.
 */
bool extractOptionalFlag(
    PropertySet::Ptr const & properties,
    std::string      const & name
) {
    if (properties && properties->exists(name)) {
        return properties->getAsBool(name);
    }
    return false;
}


/**
 * Extracts the string-valued parameter with the given name from the specified policy. If the provided
 * policy pointer is null or contains no such parameter, the given default string is returned instead.
 */
std::string const extractPolicyString(
    Policy::Ptr const & policy,
    std::string const & key,
    std::string const & def
) {
    if (policy) {
        return policy->getString(key, def);
    }
    return def;
}


static char const * const sDefaultVisitNamePat      = "_tmp_visit%1%_";
static char const * const sDefaultVisitSliceNamePat = "_tmp_visit%1%_slice%2%_";


/**
 * Returns the name of the table that a single slice of a pipeline involved in the processing
 * of a single visit should use for persistence of a particular output. All slices can be
 * configured to use the same (per-visit) table name using policy parameters.
 *
 * @param[in] policy   The @c Policy containing the table name patterns from which the
 *                     the actual table name is derived. One of two possible keys is
 * expected; the first is necessary when a single per-visit table is being used, the
 * second is necessary when each pipeline slice should send output to a seperate
 * table (e.g. to avoid write contention). In what follows, @c ${itemName} refers
 * to the value of a property named @c "itemName" extracted from @a properties.
 * <dl>
 * <dt> <tt>"${itemName}.perVisitTableNamePattern"</tt> </dt>
 * <dd> A @c boost::format compatible pattern string taking a single parameter: an id for
 *      the current visit. The default is @c "_tmp_visit%1%_${itemName}". </dd>
 * <dt> <tt>"${itemName}.perSliceAndVisitTableNamePattern"</tt> </dt>
 * <dd> A @c boost::format compatible pattern string taking two parameters: an id for the
 *      current visit followed by an id for the current slice is passed to the format. The
 *      default is @c "_tmp_visit%1%_slice%2%_${itemName}" </dd>
 * </dl>
 *
 * @param[in] properties   Provides runtime specific properties necessary to construct the
 *                         output table name. The @c "itemName" property must be present and
 * set to a non empty string. The @c "visitId" property must also be present and should
 * uniquely identify the current LSST visit. If the @c "${itemName}.isPerSliceTable" property 
 * is present, is of type @c bool and is set to @c true, and a @c "sliceId" property exists,
 * then each slice will output to a seperate per-visit table.
 */
std::string const getVisitSliceTableName(
    Policy::Ptr      const & policy,
    PropertySet::Ptr const & properties
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
            sDefaultVisitSliceNamePat + itemName
        ));
        fmt % visitId % sliceId;
    } else {
        fmt.parse(extractPolicyString(
            policy,
            itemName + ".perVisitTableNamePattern",
            sDefaultVisitNamePat + itemName
        ));
        fmt % visitId;
    }
    return fmt.str();
}


/**
 * Stores the name of the table that each slice of a pipeline involved in processing a visit
 * used for persistence of its outputs. If slices were configured to all use the same (per-visit)
 * table name, a single name is stored.
 *
 * @param[out] names   The vector to store table names in.
 *
 * @param[in] policy   The Policy containing the table name patterns from which actual table
 *                     names are derived. A key named @c "${itemName}.perVisitTableNamePattern"
 *                     or @c "${itemName}.perSliceAndVisitTableNamePattern" is expected, where
 *                     @c ${itemName} refers to the value of a property named @c "itemName" extracted
 *                     from @a properties. See the documentation for getVisitSliceTableName()
 *                     for details.
 *
 * @param[in] properties   The runtime specific properties necessary to construct the retrieve
 *                         table names. The @c "itemName" property must be present and set to a non-empty
 * string. The @c "visitId" property must also be present, and shall be a non-negative integer of type
 * @c int64_t uniquely identifying the current LSST visit. If the @c "${itemName}.isPerSliceTable"
 * property is present, is of type @c bool and is set to @c true, then it is assumed that
 * @c "${itemName}.numSlices" (a positive integer of type @c int) output tables exist and
 * are to be read in.
 *
 * @sa getVisitSliceTableName()
 */
void getAllVisitSliceTableNames(
    std::vector<std::string> & names,
    Policy::Ptr        const & policy,
    PropertySet::Ptr   const & properties
) {
    std::string itemName(getItemName(properties));
    int64_t visitId         = extractVisitId(properties);
    bool    isPerSliceTable = extractOptionalFlag(properties, itemName + ".isPerSliceTable");

    boost::format fmt;
    fmt.exceptions(boost::io::all_error_bits);

    if (isPerSliceTable) {
        int numSlices = properties->getAsInt(itemName + ".numSlices");
        if (numSlices <= 0) {
            throw LSST_EXCEPT(ex::RuntimeErrorException,
                              itemName + " \".numSlices\" property value is non-positive");
        }
        fmt.parse(extractPolicyString(
            policy,
            itemName + ".perSliceAndVisitTableNamePattern",
            sDefaultVisitSliceNamePat + itemName
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
            sDefaultVisitNamePat + itemName
        ));
        fmt % visitId;
        names.push_back(fmt.str());
    }
}


/**
 * Creates the per visit and slice table identified by calling getVisitTableName() with
 * the given @a policy and @a properties. If @a policy contains a key named 
 * @c "${itemName}.templateTableName" (where where @c ${itemName} refers to the value of a property
 * named @c "itemName" extracted from @a properties), then the value of the key is used as creation
 * template. Otherwise, the template table is assumed to be named @c "${itemName}Template". Note that
 * the template table must exist in the database identified by @a location, and that if the desired
 * table already exists, an exception is thrown.
 */
void createVisitSliceTable(
    lsst::daf::persistence::LogicalLocation const & location,
    lsst::pex::policy::Policy::Ptr const & policy,
    PropertySet::Ptr const & properties
) {
    std::string itemName(getItemName(properties));
    std::string name(getVisitSliceTableName(policy, properties));
    std::string model = extractPolicyString(policy, itemName + ".templateTableName", "tmpl_" + itemName);

    lsst::daf::persistence::DbTsvStorage db;
    db.setPersistLocation(location);
    db.createTableFromTemplate(name, model);
}


/** Drops the database table(s) identified by getAllVisitSliceTables(). */
void dropAllVisitSliceTables(
    lsst::daf::persistence::LogicalLocation const & location,
    lsst::pex::policy::Policy::Ptr const & policy,
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


std::string const formatFitsProperties(lsst::daf::base::PropertySet::Ptr prop) {
    typedef std::vector<std::string> NameList;
    std::ostringstream sout;

    NameList paramNames = prop->paramNames(false);

    for (NameList::const_iterator i = paramNames.begin(), end = paramNames.end(); i != end; ++i) {
       std::size_t lastPeriod = i->rfind(char('.'));
       std::string name = (lastPeriod == std::string::npos) ? *i : i->substr(lastPeriod + 1);
       std::type_info const & type = prop->typeOf(*i);
       if (type == typeid(int)) {
           sout << boost::format("%-8s= %20d%50s") % name % prop->get<int>(*i) % "";
       } else if (type == typeid(double)) {
           sout << boost::format("%-8s= %20.15g%50s") % name % prop->get<double>(*i) % "";
       } else if (type == typeid(std::string)) {
           sout << boost::format("%-8s= '%-67s' ") % name % prop->get<std::string>(*i);
       } 
    }

    return sout.str();
}


int countFitsHeaderCards(lsst::daf::base::PropertySet::Ptr prop) {
    return prop->paramNames(false).size();
}


}}} // namespace lsst::afw::formatters
