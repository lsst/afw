// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   Filter.h
//! \brief  Class encapsulating an identifier for an LSST filter.
//
//##====----------------                                ----------------====##/

#ifndef LSST_FW_IMAGE_FILTER_H
#define LSST_FW_IMAGE_FILTER_H

#include <cassert>
#include <string>

#include <lsst/mwi/persistence/LogicalLocation.h>
#include <lsst/mwi/persistence/DbStorage.h>


namespace lsst {
namespace fw {


/*!
    \brief  Holds an integer identifier for an LSST filter.

    Currently uses a table named \b Filter to map between names and integer identifiers for a filter.
    The \b Filter table is part of the LSST Database Schema.
 */
class Filter {

public :

    enum { U = 0, G, R, I, Z, Y, NUM_FILTERS };

    Filter() : _id(U) {}

    /*! Creates a Filter with the given identifier. Implicit conversions from \c int are allowed. */
    Filter(int id) : _id(id) { assert(id >= U && id < NUM_FILTERS); }

    /*!
        Creates a Filter with the given name, using the \b Filter table in the database currently
        set on the given DbStorage to map the filter name to an integer identifier.
     */
    Filter(lsst::mwi::persistence::DbStorage & db, std::string const & name) : _id(nameToId(db, name)) {}

    Filter(lsst::mwi::persistence::LogicalLocation const & location, std::string const & name);

    operator int() const { return _id; }
    int getId()    const { return _id; }

    std::string const toString(lsst::mwi::persistence::DbStorage & db);
    std::string const toString(lsst::mwi::persistence::LogicalLocation const & location);

private :

    int _id;

    static int nameToId(lsst::mwi::persistence::DbStorage & db, std::string const & name);
};


}}  // end of namespace lsst::fw

#endif // LSST_FW_IMAGE_FILTER_H
