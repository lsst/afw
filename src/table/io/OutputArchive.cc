// -*- lsst-c++ -*-

#include <typeinfo>
#include <vector>
#include <map>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/ArchiveIndexSchema.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

ArchiveIndexSchema const & indexKeys = ArchiveIndexSchema::get();

typedef std::map<void const *,int> Map;

typedef Map::value_type MapItem;

} // anonymous

// ----- OutputArchive::Impl --------------------------------------------------------------------------------

class OutputArchive::Impl {
public:

    BaseCatalog makeCatalog(Schema const & schema) {
        int catArchive = 1;
        CatalogVector::iterator iter = _catalogs.begin();
        int const flags = table::Schema::EQUAL_KEYS | table::Schema::EQUAL_NAMES;
        for (; iter != _catalogs.end(); ++iter, ++catArchive) {
            if (iter->getSchema().compare(schema, flags) == flags) {
                break;
            }
        }
        if (iter == _catalogs.end()) {
            iter = _catalogs.insert(_catalogs.end(), BaseCatalog(schema));
        }
        if (!iter->getTable()->getMetadata()) {
            PTR(daf::base::PropertyList) metadata(new daf::base::PropertyList());
            iter->getTable()->setMetadata(metadata);
            metadata->set("EXTTYPE", "ARCHIVE_DATA");
            metadata->set("AR_CATN", catArchive, "# of this catalog relative to the start of this archive");
        }
        return BaseCatalog(iter->getTable());
    }

    void saveCatalog(
        BaseCatalog const & catalog, int id,
        std::string const & name, std::string const & module, 
        int catPersistable
    ) {
        PTR(BaseRecord) indexRecord = _index.addNew();
        indexRecord->set(indexKeys.id, id);
        indexRecord->set(indexKeys.name, name);
        indexRecord->set(indexKeys.module, module);
        indexRecord->set(indexKeys.catPersistable, catPersistable);
        indexRecord->set(indexKeys.nRows, catalog.size());
        int catArchive = 1;
        CatalogVector::iterator iter = _catalogs.begin();
        for (; iter != _catalogs.end(); ++iter, ++catArchive) {
            if (iter->getTable() == catalog.getTable()) {
                break;
            }
        }
        if (iter == _catalogs.end()) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                "All catalogs passed to saveCatalog must be created by makeCatalog"
            );
        }
        iter->getTable()->getMetadata()->add("AR_NAME", name, "Class name for objects stored here");
        indexRecord->set(indexKeys.row0, iter->size());
        indexRecord->set(indexKeys.catArchive, catArchive);
        iter->insert(iter->end(), catalog.begin(), catalog.end(), false);
    }

    int put(Persistable const * obj, PTR(Impl) const & self, bool permissive) {
        if (!obj) return 0;
        if (permissive && !obj->isPersistable()) return 0;
        MapItem item(obj, _nextId);
        std::pair<Map::iterator,bool> r = _map.insert(item);
        if (r.second) {
            ++_nextId;
            OutputArchiveHandle handle(
                r.first->second, obj->getPersistenceName(), obj->getPythonModule(), self
            );
            obj->write(handle);
        }
        assert(r.first->first == obj);
        // either way we return the ID of the object in the archive
        return r.first->second;
    }

    void writeFits(fits::Fits & fitsfile) {
        _index.getTable()->getMetadata()->set("AR_NCAT", int(_catalogs.size() + 1),
                                              "# of catalogs in this archive, including the index");
        _index.writeFits(fitsfile);
        int n = 1;
        for (
            CatalogVector::const_iterator iter = _catalogs.begin();
            iter != _catalogs.end();
            ++iter, ++n
        ) {
            iter->writeFits(fitsfile);
        }
    }
    
    Impl() : _nextId(1), _map(), _index(ArchiveIndexSchema::get().schema) {
        PTR(daf::base::PropertyList) metadata(new daf::base::PropertyList());
        metadata->set("EXTTYPE", "ARCHIVE_INDEX");
        metadata->set("AR_CATN", 0, "# of this catalog relative to the start of this archive");
        _index.getTable()->setMetadata(metadata);
    }
    
    int _nextId;
    Map _map;
    BaseCatalog _index;
    CatalogVector _catalogs;
};

// ----- OutputArchive --------------------------------------------------------------------------------------

OutputArchive::OutputArchive() : _impl(new Impl()) {}

OutputArchive::OutputArchive(OutputArchive const & other) : _impl(other._impl) {}

OutputArchive & OutputArchive::operator=(OutputArchive const & other) {
    _impl = other._impl;
    return *this;
}

OutputArchive::~OutputArchive() {}

int OutputArchive::put(Persistable const * obj, bool permissive) {
    if (!_impl.unique()) { // copy on write
        PTR(Impl) tmp(new Impl(*_impl));
        _impl.swap(tmp);
    }
    return _impl->put(obj, _impl, permissive);
}

BaseCatalog const & OutputArchive::getIndexCatalog() const {
    return _impl->_index;
}

BaseCatalog const & OutputArchive::getCatalog(int n) const {
    if (n == 0) return _impl->_index;
    if (std::size_t(n) > _impl->_catalogs.size() || n < 0) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthErrorException,
            (boost::format("Catalog number %d is out of range [0,%d]") % n % _impl->_catalogs.size()).str()
        );
    }
    return _impl->_catalogs[n-1];
}

int OutputArchive::countCatalogs() const { return _impl->_catalogs.size() + 1; }

void OutputArchive::writeFits(fits::Fits & fitsfile) const {
    _impl->writeFits(fitsfile);
}

// ----- OutputArchiveHandle ------------------------------------------------------------------------------

BaseCatalog OutputArchiveHandle::makeCatalog(Schema const & schema) {
    return _impl->makeCatalog(schema);
}

void OutputArchiveHandle::saveCatalog(BaseCatalog const & catalog) {
    _impl->saveCatalog(catalog, _id, _name, _module, _catPersistable);
    ++_catPersistable;
}

int OutputArchiveHandle::put(Persistable const * obj, bool permissive) {
    // Handle doesn't worry about copy-on-write, because Handles should only exist
    // while an OutputArchive::put() call is active.
    return _impl->put(obj, _impl, permissive);
}

OutputArchiveHandle::OutputArchiveHandle(
    int id, std::string const & name, std::string const & module,
    PTR(OutputArchive::Impl) impl) :
    _id(id), _catPersistable(0), _name(name), _module(module), _impl(impl)
{}

OutputArchiveHandle::~OutputArchiveHandle() {}

}}}} // namespace lsst::afw::table::io
