// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   Filter.cc
//! \brief  Implements looking up a filter identifier by name.
//
//##====----------------                                ----------------====##/

#include <lsst/daf/persistence/Persistence.h>
#include <lsst/daf/persistence/Storage.h>
#include <lsst/pex/policy/Policy.h>
#include <lsst/pex/exceptions.h>

#include <lsst/afw/image/Filter.h>


using namespace lsst::afw::image;

using lsst::daf::persistence::Persistence;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;


/*!
    Creates a Filter with the given name, using the \c Filter table in the database given by
    \a location to map the filter name to an integer identifier.
 */
Filter::Filter(lsst::daf::persistence::LogicalLocation const & location, std::string const & name) {
    Policy::Ptr      noPolicy;
    Persistence::Ptr persistence = Persistence::getPersistence(noPolicy);
    Storage::Ptr     storage     = persistence->getRetrieveStorage("DbStorage", location);
    lsst::daf::persistence::DbStorage * db = dynamic_cast<lsst::daf::persistence::DbStorage *>(storage.get());
    if (db == 0) {
        throw lsst::pex::exceptions::Runtime("Didn't get DbStorage");
    }
    db->startTransaction();
    try {
        _id = nameToId(*db, name);
    } catch(...) {
        db->endTransaction();
        throw;
    }
    db->endTransaction();
}


/*!
    Returns the name of the filter, using the \b Filter table in the database given by
    \a location to map the filter identifier to a name.
 */
std::string const Filter::toString(lsst::daf::persistence::LogicalLocation const & location) {
    Policy::Ptr      noPolicy;
    Persistence::Ptr persistence = Persistence::getPersistence(noPolicy);
    Storage::Ptr     storage     = persistence->getRetrieveStorage("DbStorage", location);
    lsst::daf::persistence::DbStorage * db = dynamic_cast<lsst::daf::persistence::DbStorage *>(storage.get());
    if (db == 0) {
        throw lsst::pex::exceptions::Runtime("Didn't get DbStorage");
    }
    db->startTransaction();
    std::string result;
    try {
        result = toString(*db);
    } catch(...) {
        db->endTransaction();
        throw;
    }
    db->endTransaction();
    return result;
}


/*!
    Returns the name of the filter, using the \b Filter table in the database currently
    set on the given DbStorage to map the filter identifier to a name.
 */
std::string const Filter::toString(lsst::daf::persistence::DbStorage & db) {
    db.setTableForQuery("Filter");
    db.outColumn("filtName");
    // CORAL always maps MYSQL_TYPE_LONG (MySQL internal type specifier for INTEGER columns) to long
    db.condParam<long>("id", static_cast<long>(_id));
    db.setQueryWhere("filterId = :id");
    db.query();
    // ScopeGuard g(boost::bind(&lsst::daf::persistence::DbStorage::finishQuery, &db));
    std::string filterName;
    try {
        if (!db.next() || db.columnIsNull(0)) {
            throw lsst::pex::exceptions::Runtime("Failed to get name for filter " + _id);
        }
        filterName = db.getColumnByPos<std::string>(0);
        if (db.next()) {
            throw lsst::pex::exceptions::Runtime("Multiple names for filter " + _id);
        }
    } catch(...) {
        db.finishQuery();
        throw;
    }
    db.finishQuery();
    return filterName;
}


int Filter::nameToId(lsst::daf::persistence::DbStorage & db, std::string const & name) {
    db.setTableForQuery("Filter");
    db.outColumn("filterId");
    db.condParam<std::string>("name", name);
    db.setQueryWhere("filtName = :name");
    db.query();
    int filterId;
    try {
        if (!db.next() || db.columnIsNull(0)) {
            throw lsst::pex::exceptions::Runtime("Failed to get id for filter named " + name);
        }
        filterId = static_cast<int>(db.getColumnByPos<long>(0));
        if (db.next()) {
            throw lsst::pex::exceptions::Runtime("Multiple ids for filter named " + name);
        }
        if (filterId < U || filterId >= NUM_FILTERS) {
            throw lsst::pex::exceptions::OutOfRange("Invalid filter id for filter named " + name);
        }
    } catch (...) {
        db.finishQuery();
        throw;
    }
    db.finishQuery();
    return filterId;
}
