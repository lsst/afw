// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

//
//##====----------------                                ----------------====##/
//
//         Implements looking up a filter identifier by name.
//
//##====----------------                                ----------------====##/
#include <cmath>
#include "boost/format.hpp"
#include "boost/algorithm/string/trim.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base/PropertySet.h"

#include "lsst/afw/image/Filter.h"

#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

namespace pexEx = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace image {

FilterProperty::PropertyMap* FilterProperty::_propertyMap = NULL;

FilterProperty::FilterProperty(std::string const& name, lsst::daf::base::PropertySet const& prop, bool force)
        : _name(name), _lambdaEff(NAN), _lambdaMin(NAN), _lambdaMax(NAN) {
    if (prop.exists("lambdaEff")) {
        _lambdaEff = prop.getAsDouble("lambdaEff");
    }
    if (prop.exists("lambdaMin")) {
        _lambdaMin = prop.getAsDouble("lambdaMin");
    }
    if (prop.exists("lambdaMax")) {
        _lambdaMax = prop.getAsDouble("lambdaMax");
    }
    _insert(force);
}

void FilterProperty::_insert(bool force) {
    if (!_propertyMap) {
        _initRegistry();
    }

    PropertyMap::iterator keyVal = _propertyMap->find(getName());

    if (keyVal != _propertyMap->end()) {
        if (keyVal->second == *this) {
            return;  // OK, a redefinition with identical values
        }

        if (!force) {
            throw LSST_EXCEPT(pexEx::RuntimeError, "Filter " + getName() + " is already defined");
        }
        _propertyMap->erase(keyVal);
    }

    _propertyMap->insert(std::make_pair(getName(), *this));
}

bool FilterProperty::operator==(FilterProperty const& rhs) const noexcept {
    return (_lambdaEff == rhs._lambdaEff);
}

std::size_t FilterProperty::hash_value() const noexcept { return std::hash<double>()(_lambdaEff); };

void FilterProperty::_initRegistry() {
    if (_propertyMap) {
        delete _propertyMap;
    }

    _propertyMap = new PropertyMap;
}

FilterProperty const& FilterProperty::lookup(std::string const& name) {
    if (!_propertyMap) {
        _initRegistry();
    }

    PropertyMap::iterator keyVal = _propertyMap->find(name);

    if (keyVal == _propertyMap->end()) {
        throw LSST_EXCEPT(pexEx::NotFoundError, "Unable to find filter " + name);
    }

    return keyVal->second;
}

namespace {
std::string const unknownFilter = "_unknown_";
}

int const Filter::AUTO = -1;
int const Filter::UNKNOWN = -1;

Filter::Filter(std::shared_ptr<lsst::daf::base::PropertySet const> metadata, bool const force) {
    std::string const key = "FILTER";
    if (metadata->exists(key)) {
        std::string filterName = boost::algorithm::trim_right_copy(metadata->getAsString(key));
        _id = _lookup(filterName, force);
        _name = filterName;
    }
}

namespace detail {
int stripFilterKeywords(std::shared_ptr<lsst::daf::base::PropertySet> metadata) {
    int nstripped = 0;

    std::string key = "FILTER";
    if (metadata->exists(key)) {
        metadata->remove(key);
        nstripped++;
    }

    return nstripped;
}
}  // namespace detail

// N.b. we cannot declare a std::vector<std::string const&> as there's no way to push the references
std::vector<std::string> Filter::getAliases() const {
    std::vector<std::string> aliases;

    std::string const& canonicalName = getCanonicalName();
    for (AliasMap::iterator ptr = _aliasMap->begin(), end = _aliasMap->end(); ptr != end; ++ptr) {
        if (ptr->second == canonicalName) {
            aliases.push_back(ptr->first);
        }
    }

    return aliases;
}

std::vector<std::string> Filter::getNames() {
    if (!_nameMap) {
        _initRegistry();
    }

    std::vector<std::string> names;

    for (NameMap::const_iterator ptr = _nameMap->begin(), end = _nameMap->end(); ptr != end; ++ptr) {
        if (ptr->first != unknownFilter) {
            names.push_back(ptr->first);
        }
    }
    std::sort(names.begin(), names.end());

    return names;
}

std::shared_ptr<typehandling::Storable> Filter::cloneStorable() const {
    return std::make_unique<Filter>(*this);
}

bool Filter::equals(typehandling::Storable const& other) const noexcept {
    return singleClassEquals(*this, other);
}

namespace {

struct PersistenceHelper {
    table::Schema schema;
    table::Key<std::string> name;

    static PersistenceHelper const& get() {
        static PersistenceHelper const instance;
        return instance;
    }

private:
    PersistenceHelper() : schema(), name(schema.addField<std::string>("name", "name of the filter")) {
        schema.getCitizen().markPersistent();
    }
};

class FilterFactory : public table::io::PersistableFactory {
public:
    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        PersistenceHelper const& keys = PersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        return std::make_shared<Filter>(catalogs.front().begin()->get(keys.name), true);
    }

    FilterFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

std::string getPersistenceName() { return "Filter"; }

FilterFactory registration(getPersistenceName());

}  // namespace

bool Filter::isPersistable() const noexcept { return true; }

std::string Filter::getPersistenceName() const { return getPersistenceName(); }

std::string Filter::getPythonModule() const { return "lsst.afw.image"; };

void Filter::write(OutputArchiveHandle& handle) const {
    PersistenceHelper const& keys = PersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = catalog.addNew();
    record->set(keys.name, getName());
    handle.saveCatalog(catalog);
}

bool Filter::operator==(Filter const& rhs) const noexcept { return _id != UNKNOWN && _id == rhs._id; }

std::size_t Filter::hash_value() const noexcept { return std::hash<int>()(_id); }

void Filter::_initRegistry() {
    _id0 = UNKNOWN;
    delete _aliasMap;
    delete _nameMap;
    delete _idMap;

    _aliasMap = new AliasMap;
    _nameMap = new NameMap;
    _idMap = new IdMap;

    define(FilterProperty(unknownFilter, daf::base::PropertySet(), true));
}

int Filter::_id0 = Filter::UNKNOWN;

// dynamically allocated as that avoids an intel bug with static variables in dynamic libraries
Filter::AliasMap* Filter::_aliasMap = NULL;
Filter::NameMap* Filter::_nameMap = NULL;
Filter::IdMap* Filter::_idMap = NULL;

int Filter::define(FilterProperty const& fp, int id, bool force) {
    if (!_nameMap) {
        _initRegistry();
    }

    std::string const& name = fp.getName();
    NameMap::iterator keyVal = _nameMap->find(name);

    if (keyVal != _nameMap->end()) {
        int oid = keyVal->second;

        if (id == oid || id == AUTO) {
            return oid;  // OK, same value as before
        }

        if (!force) {
            throw LSST_EXCEPT(pexEx::RuntimeError, "Filter " + name + " is already defined");
        }
        _nameMap->erase(keyVal);
        _idMap->erase(oid);
    }

    if (id == AUTO) {
        id = _id0;
        ++_id0;
    }

    _nameMap->insert(std::make_pair(name, id));
    _idMap->insert(std::make_pair(id, name));

    return id;
}

int Filter::defineAlias(std::string const& oldName, std::string const& newName, bool force) {
    if (!_nameMap) {
        _initRegistry();
    }

    // Lookup oldName
    NameMap::iterator keyVal = _nameMap->find(oldName);
    if (keyVal == _nameMap->end()) {
        throw LSST_EXCEPT(pexEx::NotFoundError, "Unable to find filter " + oldName);
    }
    int const id = keyVal->second;

    // Lookup oldName in aliasMap
    AliasMap::iterator aliasKeyVal = _aliasMap->find(newName);
    if (aliasKeyVal != _aliasMap->end()) {
        if (aliasKeyVal->second == oldName) {
            return id;  // OK, same value as before
        }

        if (!force) {
            throw LSST_EXCEPT(pexEx::NotFoundError, "Filter " + newName + " is already defined");
        }
        _aliasMap->erase(aliasKeyVal);
    }

    _aliasMap->insert(std::make_pair(newName, oldName));

    return id;
}

int Filter::_lookup(std::string const& name, bool const force) {
    if (!_nameMap) {
        _initRegistry();
    }

    NameMap::iterator keyVal = _nameMap->find(name);

    if (keyVal == _nameMap->end()) {
        AliasMap::iterator aliasKeyVal = _aliasMap->find(name);
        if (aliasKeyVal != _aliasMap->end()) {
            return _lookup(aliasKeyVal->second);
        }

        if (force) {
            return UNKNOWN;
        } else {
            throw LSST_EXCEPT(pexEx::NotFoundError, "Unable to find filter " + name);
        }
    }

    return keyVal->second;
}

std::string const& Filter::_lookup(int id) {
    if (!_idMap) {
        _initRegistry();
    }

    IdMap::iterator keyVal = _idMap->find(id);

    if (keyVal == _idMap->end()) {
        throw LSST_EXCEPT(pexEx::NotFoundError, (boost::format("Unable to find filter %d") % id).str());
    }

    return keyVal->second;
}
FilterProperty const& Filter::getFilterProperty() const {
    //
    // Map name to its ID and back to resolve aliases
    //
    int const id = _lookup(_name, true);
    std::string const& name = (id == UNKNOWN) ? _name : _lookup(id);

    return FilterProperty::lookup(name);
}
}  // namespace image
}  // namespace afw
}  // namespace lsst
