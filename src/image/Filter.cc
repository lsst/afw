// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Implements looking up a filter identifier by name.
//
//##====----------------                                ----------------====##/

#include "boost/format.hpp"
#include "lsst/pex/exceptions.h"

#include "lsst/afw/image/Filter.h"

namespace pexEx = lsst::pex::exceptions;

namespace lsst { namespace afw { namespace image {

FilterProperty::NameMap *FilterProperty::_nameMap = NULL;

FilterProperty::FilterProperty(std::string const& name, ///< name of filter
                               double lambdaEff, ///< Effective wavelength (nm)
                               bool force        ///< Allow this name to replace a previous one
                              ) : _name(name), _lambdaEff(lambdaEff)
{
    if (!_nameMap) {
        _initRegistry();
    }

    NameMap::iterator keyVal = _nameMap->find(name);

    if (keyVal != _nameMap->end()) {
        if (!force) {
            throw LSST_EXCEPT(pexEx::RuntimeErrorException, "Filter " + name + " is already defined");
        }
        _nameMap->erase(keyVal);
    }
    
    _nameMap->insert(std::make_pair(name, *this));
}
            
/**
 * Initialise the Filter registry
 */
void FilterProperty::_initRegistry()
{
    if (_nameMap) {
        delete _nameMap;
    }

    _nameMap = new NameMap;
}

/**
 * Lookup the properties of a filter "name"
 */
FilterProperty const& FilterProperty::lookup(std::string const& name ///< name of desired filter
                                            )
{
    if (!_nameMap) {
        _initRegistry();
    }

    NameMap::iterator keyVal = _nameMap->find(name);

    if (keyVal == _nameMap->end()) {
        throw LSST_EXCEPT(pexEx::NotFoundException, "Unable to find filter " + name);
    }
    
    return keyVal->second;
}
            
/************************************************************************************************************/
/**
 * Initialise the Filter registry
 */
void Filter::_initRegistry()
{
    _id0 = UNKNOWN;
    delete _nameMap;
    delete _idMap;

    _nameMap = new NameMap;
    _idMap = new IdMap;
    
    define(FilterProperty("_unknown_", -1));
}

/************************************************************************************************************/

int Filter::_id0 = Filter::UNKNOWN;

Filter::NameMap *Filter::_nameMap = NULL; // dynamically allocated as that avoids an intel bug with static
                                        // variables in dynamic libraries
Filter::IdMap *Filter::_idMap = NULL; // dynamically allocated as that avoids an intel bug with static
                                        // variables in dynamic libraries

/**
 * Define a filter name to have the specified id
 *
 * If id == Filter::AUTO a value will be chosen for you.
 *
 * It is an error to attempt to change a name's id (unless you specify force)
 */
int Filter::define(FilterProperty const& fp, int id, bool force)
{
    if (!_nameMap) {
        _initRegistry();
    }

    std::string const& name = fp.getName();
    NameMap::iterator keyVal = _nameMap->find(name);

    if (id == AUTO) {
        id = _id0;
        ++_id0;
    }
    
    if (keyVal != _nameMap->end()) {
        int oid = keyVal->second;

        if (oid == id) {
            return id;                  // OK, same value as before
        }

        if (!force) {
            throw LSST_EXCEPT(pexEx::RuntimeErrorException, "Filter " + name + " is already defined");
        }
        _nameMap->erase(keyVal);
        _idMap->erase(oid);
    }
    
    _nameMap->insert(std::make_pair(name, id));
    _idMap->insert(std::make_pair(id, name));

    return id;
}

/**
 * Lookup the ID associated with a name
 */
int Filter::_lookup(std::string const& name)
{
    if (!_nameMap) {
        _initRegistry();
    }

    NameMap::iterator keyVal = _nameMap->find(name);

    if (keyVal == _nameMap->end()) {
        throw LSST_EXCEPT(pexEx::NotFoundException, "Unable to find filter " + name);
    }
    
    return keyVal->second;
}

/**
 * Lookup the name associated with an ID
 */
std::string const& Filter::_lookup(int id)
{
    if (!_idMap) {
        _initRegistry();
    }

    IdMap::iterator keyVal = _idMap->find(id);

    if (keyVal == _idMap->end()) {
        throw LSST_EXCEPT(pexEx::NotFoundException, (boost::format("Unable to find filter %d") % id).str());
    }
    
    return keyVal->second;
}
            
}}}
