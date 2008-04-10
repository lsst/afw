// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Class encapsulating an identifier for an LSST filter.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_IMAGE_FILTER_H
#define LSST_AFW_IMAGE_FILTER_H

#include <cassert>
#include <string>

#include <lsst/daf/persistence/LogicalLocation.h>
#include <lsst/daf/persistence/DbStorage.h>


namespace lsst {
namespace afw {
namespace image {

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
    Filter(lsst::daf::persistence::DbStorage & db, std::string const & name) : _id(nameToId(db, name)) {}

    Filter(lsst::daf::persistence::LogicalLocation const & location, std::string const & name);

    operator int() const { return _id; }
    int getId()    const { return _id; }

    std::string const toString(lsst::daf::persistence::DbStorage & db);
    std::string const toString(lsst::daf::persistence::LogicalLocation const & location);

private :

    int _id;

    static int nameToId(lsst::daf::persistence::DbStorage & db, std::string const & name);
};

}}}  // lsst::afw::image

#endif // LSST_AFW_IMAGE_FILTER_H
