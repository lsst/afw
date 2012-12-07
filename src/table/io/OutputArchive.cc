// -*- lsst-c++ -*-

#include <typeinfo>
#include <vector>
#include <map>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/ArchiveIndexSchema.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

ArchiveIndexSchema const & indexKeys = ArchiveIndexSchema::get();

typedef std::map<void const *,int> Map;

typedef Map::value_type MapItem;

} // anonymous

// ----- OutputArchive::Impl --------------------------------------------------------------------------------

struct OutputArchive::Impl {

    PTR(BaseRecord) addCatalog(Schema const & schema, int id, std::string const & name, int catPersistable) {
        PTR(BaseRecord) indexRecord = _index.addNew();
        indexRecord->set(indexKeys.id, id);
        indexRecord->set(indexKeys.name, name);
        indexRecord->set(indexKeys.catPersistable, catPersistable);
        indexRecord->set(indexKeys.nRows, 0);
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
        iter->getTable()->getMetadata()->add("AR_NAME", name, "Class name for objects stored here");
        indexRecord->set(indexKeys.row0, iter->size());
        indexRecord->set(indexKeys.catArchive, catArchive);
        return indexRecord;
    }

    PTR(BaseRecord) addRecord(BaseRecord & indexRecord) {
        BaseCatalog & catalog = _catalogs[indexRecord.get(indexKeys.catArchive)-1];
        // check to make sure the block of rows hasn't been interrupted by a row from another object.
        if (int(catalog.size()) != indexRecord.get(indexKeys.row0) + indexRecord.get(indexKeys.nRows)) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                "Logic error in nested persistence: classes should not alternate calls to 'put' "
                "and 'addRecord' if a nested object may reuse the parent object's schema."
            );
        }
        ++indexRecord[indexKeys.nRows];
        return catalog.addNew();
    }

    int put(Persistable const * obj, PTR(Impl) const & self) {
        if (!obj) return 0;
        MapItem item(obj, _nextId);
        std::pair<Map::iterator,bool> r = _map.insert(item);
        if (r.second) {
            // insertion successful, which means it's a new object and we should tell it to save itself now
            try {
                ++_nextId;
                Handle handle(r.first->second, obj->getPersistenceName(), self);
                obj->write(handle);
            } catch (...) {
                --_nextId;
                _map.erase(r.first);
                throw;
            }
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

int OutputArchive::put(Persistable const * obj) {
    if (!_impl.unique()) { // copy on write
        PTR(Impl) tmp(new Impl(*_impl));
        _impl.swap(tmp);
    }
    return _impl->put(obj, _impl);
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

// ----- OutputArchive::CatalogProxy ------------------------------------------------------------------------

PTR(BaseRecord) OutputArchive::CatalogProxy::addRecord() {
    if (!_index) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Cannot create a new record without first creating a catalog."
        );
    }
    return _impl->addRecord(*_index);
}

OutputArchive::CatalogProxy::CatalogProxy(PTR(BaseRecord) index, PTR(Impl) impl) :
    _index(index), _impl(impl)
{}

OutputArchive::CatalogProxy::CatalogProxy(CatalogProxy const & other) :
    _index(other._index), _impl(other._impl)
{}

OutputArchive::CatalogProxy & OutputArchive::CatalogProxy::operator=(CatalogProxy const & other) {
    _index = other._index;
    _impl = other._impl;
    return *this;
}

OutputArchive::CatalogProxy::~CatalogProxy() {}

// ----- OutputArchive::Handle ------------------------------------------------------------------------------

OutputArchive::CatalogProxy OutputArchive::Handle::addCatalog(Schema const & schema) {
    PTR(BaseRecord) index = _impl->addCatalog(schema, _id, _name, _catPersistable);
    return CatalogProxy(index, _impl);
}

int OutputArchive::Handle::put(Persistable const * obj) {
    // Handle doesn't worry about copy-on-write, because Handles should only exist
    // while an OutputArchive::put() call is active.
    return _impl->put(obj, _impl);
}

OutputArchive::Handle::Handle(int id, std::string const & name, PTR(Impl) impl) :
    _id(id), _name(name), _catPersistable(0), _impl(impl)
{}

OutputArchive::Handle::~Handle() {}

}}}} // namespace lsst::afw::table::io
