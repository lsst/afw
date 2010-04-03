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

FilterProperty::PropertyMap *FilterProperty::_propertyMap = NULL;

FilterProperty::FilterProperty(std::string const& name, ///< name of filter
                               double lambdaEff, ///< Effective wavelength (nm)
                               bool force        ///< Allow this name to replace a previous one
                              ) : _name(name), _lambdaEff(lambdaEff)
{
    if (!_propertyMap) {
        _initRegistry();
    }

    PropertyMap::iterator keyVal = _propertyMap->find(name);

    if (keyVal != _propertyMap->end()) {
        if (!force) {
            throw LSST_EXCEPT(pexEx::RuntimeErrorException, "Filter " + name + " is already defined");
        }
        _propertyMap->erase(keyVal);
    }
    
    _propertyMap->insert(std::make_pair(name, *this));
}
            
/**
 * Initialise the Filter registry
 */
void FilterProperty::_initRegistry()
{
    if (_propertyMap) {
        delete _propertyMap;
    }

    _propertyMap = new PropertyMap;
}

/**
 * Lookup the properties of a filter "name"
 */
FilterProperty const& FilterProperty::lookup(std::string const& name ///< name of desired filter
                                            )
{
    if (!_propertyMap) {
        _initRegistry();
    }

    PropertyMap::iterator keyVal = _propertyMap->find(name);

    if (keyVal == _propertyMap->end()) {
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
    delete _aliasMap;
    delete _nameMap;
    delete _idMap;

    _aliasMap = new AliasMap;
    _nameMap = new NameMap;
    _idMap = new IdMap;
    
    define(FilterProperty("_unknown_", -1, true));
}

/************************************************************************************************************/

int Filter::_id0 = Filter::UNKNOWN;

Filter::AliasMap *Filter::_aliasMap = NULL; // dynamically allocated as that avoids an intel bug with static
                                        // variables in dynamic libraries
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
 * Define an alias for a filter
 */
int Filter::defineAlias(std::string const& oldName, ///< old name for Filter
                        std::string const& newName, ///< new name for Filter
                        bool force                  ///< force an alias even if newName is already in use
                       )
{
    if (!_nameMap) {
        _initRegistry();
    }

    // Lookup oldName
    NameMap::iterator keyVal = _nameMap->find(oldName);
    if (keyVal == _nameMap->end()) {
        throw std::exception();
    }
    int const id = keyVal->second;

    // Lookup oldName in aliasMap
    AliasMap::iterator aliasKeyVal = _aliasMap->find(newName);
    if (aliasKeyVal != _aliasMap->end()) {
        if (aliasKeyVal->second == oldName) {
            return id;                  // OK, same value as before
        }

        if (!force) {
            throw std::exception(); // std::cerr << "Filter " << name << " is already defined" << std::endl;
        }
        _aliasMap->erase(aliasKeyVal);
    }
    
    _aliasMap->insert(std::make_pair(newName, oldName));

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
        AliasMap::iterator aliasKeyVal = _aliasMap->find(name);
        if (aliasKeyVal != _aliasMap->end()) {
            return _lookup(aliasKeyVal->second);
        }
        
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
