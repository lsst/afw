// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "boost/make_shared.hpp"

#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst { namespace afw { namespace image {

// Even though this static const member is set in the header, it needs to be declared here if we need
// to take its address (and Swig might).
std::size_t const ApCorrMap::MAX_NAME_LENGTH;

PTR(math::BoundedField) const ApCorrMap::operator[](std::string const & name) const {
    Iterator i = _internal.find(name);
    if (i == _internal.end()) {
        throw LSST_EXCEPT(
            pex::exceptions::NotFoundError,
            (boost::format("Aperture correction with name '%s' not found") % name).str()
        );
    }
    return i->second;
}

PTR(math::BoundedField) const ApCorrMap::get(std::string const & name) const {
    Iterator i = _internal.find(name);
    if (i == _internal.end()) {
        return PTR(math::BoundedField)();
    }
    return i->second;
}

void ApCorrMap::set(std::string const & name, PTR(math::BoundedField) field) {
    if (name.size() > MAX_NAME_LENGTH) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            (boost::format("Aperture correction name '%s' exceeds size limit of %d characters")
             % name % MAX_NAME_LENGTH).str()
        );
    }
    _internal.insert(std::make_pair(name, field));
}

namespace {

struct PersistenceHelper {
    table::Schema schema;
    table::Key<std::string> name;
    table::Key<int> field;

    static PersistenceHelper const & get() {
        static PersistenceHelper const instance;
        return instance;
    }

private:

    PersistenceHelper() :
        schema(),
        name(schema.addField<std::string>("name", "name of the aperture correction",
                                          ApCorrMap::MAX_NAME_LENGTH)),
        field(schema.addField<int>("field", "archive ID of the BoundedField object"))
    {
        schema.getCitizen().markPersistent();
    }

};

class ApCorrMapFactory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        PersistenceHelper const & keys = PersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        PTR(ApCorrMap) result
            = boost::make_shared<ApCorrMap>();
        for (
            table::BaseCatalog::const_iterator i = catalogs.front().begin();
            i != catalogs.front().end();
            ++i
        ) {
            result->set(i->get(keys.name), archive.get<math::BoundedField>(i->get(keys.field)));
        }
        return result;
    }

    ApCorrMapFactory(std::string const & name) : afw::table::io::PersistableFactory(name) {}

};

std::string getApCorrMapPersistenceName() {
    return "ApCorrMap";
}

ApCorrMapFactory registration(getApCorrMapPersistenceName());

} // anonymous

bool ApCorrMap::isPersistable() const {
    for (Iterator i = begin(); i != end(); ++i) {
        if (!i->second->isPersistable()) return false;
    }
    return true;
}

std::string ApCorrMap::getPersistenceName() const {
    return getApCorrMapPersistenceName();
}

std::string ApCorrMap::getPythonModule() const {
    return "lsst.afw.image";
}

void ApCorrMap::write(OutputArchiveHandle & handle) const {
    PersistenceHelper const & keys = PersistenceHelper::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    for (Iterator i = begin(); i != end(); ++i) {
        PTR(table::BaseRecord) record = catalog.addNew();
        record->set(keys.name, i->first);
        record->set(keys.field, handle.put(i->second));
    }
    handle.saveCatalog(catalog);
}

void ApCorrMap::operator*=(double const scale) {
    Internal replacement;
    for (Iterator i = begin(); i != end(); ++i) {
        replacement[i->first] = (*i->second)*scale;
    }
    _internal = replacement;
}

}}} // namespace lsst::afw::image
