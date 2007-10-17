// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file   DbAccess.h
//! \brief  Encapsulates CORAL initialization/cleanup, and exposes the
//!         fundamental CORAL objects needed to interact with the database
//!
//##====----------------                                ----------------====##/

#ifndef LSST_FW_FORMATTERS_DB_ACCESS_H
#define LSST_FW_FORMATTERS_DB_ACCESS_H

#include "RelationalAccess/IConnection.h"
#include "RelationalAccess/ISession.h"

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

#include <lsst/mwi/persistence/LogicalLocation.h>


namespace lsst {
namespace fw {
namespace formatters {

using lsst::mwi::persistence::LogicalLocation;


class DbAccess : private boost::noncopyable {

public :

    DbAccess(LogicalLocation const & location, bool const readAccess);

    ~DbAccess();

    coral::IConnection & connection() { return *_connection.get(); }
    coral::ISession    & session()    { return *_session.get();    }
    
private :

    boost::scoped_ptr<coral::IConnection> _connection;
    boost::scoped_ptr<coral::ISession>    _session;
};


}}} // end of namespace lsst::fw::formatters

#endif // LSST_FW_FORMATTERS_DB_ACCESS_H

