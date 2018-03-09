// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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

#include <memory>

#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst {
namespace afw {
namespace image {

// Even though this static const member is set in the header, it needs to be declared here if we need
// to take its address (and Swig might).
std::size_t const ApCorrMap::MAX_NAME_LENGTH;

std::shared_ptr<math::BoundedField> const ApCorrMap::operator[](std::string const& name) const {
    Iterator i = _internal.find(name);
    if (i == _internal.end()) {
        throw LSST_EXCEPT(pex::exceptions::NotFoundError,
                          (boost::format("Aperture correction with name '%s' not found") % name).str());
    }
    return i->second;
}

std::shared_ptr<math::BoundedField> const ApCorrMap::get(std::string const& name) const {
    Iterator i = _internal.find(name);
    if (i == _internal.end()) {
        return std::shared_ptr<math::BoundedField>();
    }
    return i->second;
}

void ApCorrMap::set(std::string const& name, std::shared_ptr<math::BoundedField> field) {
    if (name.size() > MAX_NAME_LENGTH) {
        throw LSST_EXCEPT(
                pex::exceptions::LengthError,
                (boost::format("Aperture correction name '%s' exceeds size limit of %d characters") % name %
                 MAX_NAME_LENGTH)
                        .str());
    }
    _internal.insert(std::make_pair(name, field));
}

namespace {

struct PersistenceHelper {
    table::Schema schema;
    table::Key<std::string> name;
    table::Key<int> field;

    static PersistenceHelper const& get() {
        static PersistenceHelper const instance;
        return instance;
    }

private:
    PersistenceHelper()
            : schema(),
              name(schema.addField<std::string>("name", "name of the aperture correction",
                                                ApCorrMap::MAX_NAME_LENGTH)),
              field(schema.addField<int>("field", "archive ID of the BoundedField object")) {
        schema.getCitizen().markPersistent();
    }
};

class ApCorrMapFactory : public table::io::PersistableFactory {
public:
    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                         CatalogVector const& catalogs) const override {
        PersistenceHelper const& keys = PersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        std::shared_ptr<ApCorrMap> result = std::make_shared<ApCorrMap>();
        for (table::BaseCatalog::const_iterator i = catalogs.front().begin(); i != catalogs.front().end();
             ++i) {
            result->set(i->get(keys.name), archive.get<math::BoundedField>(i->get(keys.field)));
        }
        return result;
    }

    explicit ApCorrMapFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

std::string getApCorrMapPersistenceName() { return "ApCorrMap"; }

ApCorrMapFactory registration(getApCorrMapPersistenceName());

}  // namespace

bool ApCorrMap::isPersistable() const {
    for (Iterator i = begin(); i != end(); ++i) {
        if (!i->second->isPersistable()) return false;
    }
    return true;
}

std::string ApCorrMap::getPersistenceName() const { return getApCorrMapPersistenceName(); }

std::string ApCorrMap::getPythonModule() const { return "lsst.afw.image"; }

void ApCorrMap::write(OutputArchiveHandle& handle) const {
    PersistenceHelper const& keys = PersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    for (Iterator i = begin(); i != end(); ++i) {
        std::shared_ptr<table::BaseRecord> record = catalog.addNew();
        record->set(keys.name, i->first);
        record->set(keys.field, handle.put(i->second));
    }
    handle.saveCatalog(catalog);
}

ApCorrMap& ApCorrMap::operator*=(double const scale) {
    Internal replacement;
    for (Iterator i = begin(); i != end(); ++i) {
        replacement[i->first] = (*i->second) * scale;
    }
    _internal = replacement;
    return *this;
}
}  // namespace image
}  // namespace afw
}  // namespace lsst
