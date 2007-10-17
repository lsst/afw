// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file   DbAccess.cc
//!
//##====----------------                                ----------------====##/

#include <string>
#include <vector>

#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/IRelationalService.h"
#include "RelationalAccess/RelationalServiceException.h"
#include "PluginManager/PluginManager.h"
#include "SealKernel/ComponentLoader.h"

#include <lsst/mwi/exceptions.h>
#include <lsst/mwi/persistence/DbStorage.h>
#include <lsst/mwi/persistence/DbStorageLocation.h>

#include "lsst/fw/formatters/DbAccess.h"


namespace lsst {
namespace fw {
namespace formatters {

using lsst::mwi::persistence::DbStorage;
using lsst::mwi::persistence::DbStorageLocation;


typedef std::vector<seal::IHandle<coral::IRelationalService> > RelationalServiceList;

namespace {

static void getRelationalServices(RelationalServiceList & list) {
    static seal::Handle<seal::Context> context(0);

    if (context.get() == 0) {
        context = new seal::Context;
        seal::PluginManager * pm = seal::PluginManager::get();
        pm->initialise();
        seal::Handle<seal::ComponentLoader> loader(new seal::ComponentLoader(context.get()));
        loader->load("CORAL/Services/RelationalService");
    }
    context->query(list);
}

}



DbAccess::DbAccess(LogicalLocation const & location, bool const readOnly) :
    _connection(0),
    _session(0)
{

    // Translate the logical location to a database location
    DbStorageLocation loc(location.locString());

    // Make sure that the SEAL plugin manager and CORAL services are
    // loaded by creating a DbStorage instance
    DbStorage volatile dbStorage;

    // Query for the relational service.
    RelationalServiceList list;
    getRelationalServices(list);
    if (list.empty()) {
        throw lsst::mwi::exceptions::Runtime("Unable to locate CORAL RelationalService");
    }

    // Use the connection string to get the relational domain.
    std::string connString(loc.getConnString());
    coral::IRelationalDomain & domain = list.front()->domainForConnection(connString);

    // Use the domain to decode the connection string and create a connection.
    std::pair<std::string, std::string> connAndSchema = domain.decodeUserConnectionString(connString);

    _connection.reset(domain.newConnection(connAndSchema.first));
    if (_connection.get() == 0) {
        throw lsst::mwi::exceptions::Runtime("Unable to connect to database");
    }

    // Create a session with the appropriate access mode and login.
    _session.reset(_connection->newSession(connAndSchema.second, readOnly ? coral::ReadOnly : coral::Update));
    if (_session.get() == 0) {
        throw lsst::mwi::exceptions::Runtime("Unable to start database session");
    }

    _session->startUserSession(loc.getUsername(), loc.getPassword());
    if (!_connection->isConnected()) {
        throw lsst::mwi::exceptions::Runtime("Unable to login to database");
    }
}


DbAccess::~DbAccess() {
    if (_session.get() != 0) {
        _session->endUserSession();
    }
}


}}} // end of namespace lsst::fw::formatters

