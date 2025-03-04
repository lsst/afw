// -*- lsst-c++ -*-

#include <typeinfo>
#include <vector>
#include <map>
#include <memory>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/ArchiveIndexSchema.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/fits.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

namespace {

ArchiveIndexSchema const &indexKeys = ArchiveIndexSchema::get();

// we don't need sorting, but you can't use weak_ptrs as keys in an
// unordered_map
using Map = std::map<std::weak_ptr<Persistable const>, int,
                     std::owner_less<std::weak_ptr<Persistable const>>>;

using MapItem = Map::value_type;

}  // namespace

// ----- OutputArchive::Impl --------------------------------------------------------------------------------

class OutputArchive::Impl {
public:
    BaseCatalog makeCatalog(Schema const &schema) {
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
            std::shared_ptr<daf::base::PropertyList> metadata(new daf::base::PropertyList());
            iter->getTable()->setMetadata(metadata);
            metadata->set("EXTTYPE", "ARCHIVE_DATA");
            metadata->set("AR_CATN", catArchive, "# of this catalog relative to the start of this archive");
        }
        return BaseCatalog(iter->getTable());
    }

    std::shared_ptr<BaseRecord> addIndexRecord(int id, std::string const &name, std::string const &module) {
        auto indexRecord = _index.addNew();
        indexRecord->set(indexKeys.id, id);
        indexRecord->set(indexKeys.name, name);
        indexRecord->set(indexKeys.module, module);
        return indexRecord;
    }

    void saveEmpty(int id, std::string const &name, std::string const &module) {
        auto indexRecord = addIndexRecord(id, name, module);
        indexRecord->set(indexKeys.nRows, 0);
        indexRecord->set(indexKeys.catPersistable, ArchiveIndexSchema::NO_CATALOGS_SAVED);
        indexRecord->set(indexKeys.row0, ArchiveIndexSchema::NO_CATALOGS_SAVED);
        indexRecord->set(indexKeys.catArchive, ArchiveIndexSchema::NO_CATALOGS_SAVED);
    }

    void saveCatalog(BaseCatalog const &catalog, int id, std::string const &name, std::string const &module,
                     int catPersistable) {
        auto indexRecord = addIndexRecord(id, name, module);
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
            throw LSST_EXCEPT(pex::exceptions::LogicError,
                              "All catalogs passed to saveCatalog must be created by makeCatalog");
        }
        auto metadata = iter->getTable()->getMetadata();
        // If EXTNAME exists we have already assigned an EXTVER.
        // Otherwise must scan through all catalogs looking for other
        // catalogs with the same name that have been assigned EXTNAME
        // to determine this EXTVER.
        auto found_extname = metadata->exists("EXTNAME");
        if (!found_extname) {
            int extver = 1;
            // Catalogs can be filled out of order, so look for other
            // catalogs with this EXTNAME.
            CatalogVector::iterator ver_iter = _catalogs.begin();
            for (; ver_iter != _catalogs.end(); ++ver_iter) {
                auto cat_metadata = ver_iter->getTable()->getMetadata();
                if (cat_metadata->exists("EXTNAME") && cat_metadata->getAsString("EXTNAME") == name) {
                        ++extver;
                }
            }
            if (extver > 1) { // 1 is the default so no need to write it.
                metadata->set("EXTVER", extver);
            }
        }
        // Add the name of the class to the header so anyone looking at it can
        // tell what's stored there.  But we don't want to add it multiple times.
        try {
            auto names = metadata->getArray<std::string>("AR_NAME");
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                iter->getTable()->getMetadata()->add("AR_NAME", name, "Class name for objects stored here");
            }
        } catch (pex::exceptions::NotFoundError &) {
            metadata->add("AR_NAME", name, "Class name for objects stored here");
        }
        // Also add an EXTNAME. The most recent AR_NAME given will be used.
        metadata->set("EXTNAME", name);
        indexRecord->set(indexKeys.row0, iter->size());
        indexRecord->set(indexKeys.catArchive, catArchive);
        iter->insert(iter->end(), catalog.begin(), catalog.end(), false);
    }

    int put(Persistable const *obj, std::shared_ptr<Impl> const &self, bool permissive) {
        if (!obj) return 0;
        if (permissive && !obj->isPersistable()) return 0;
        int const currentId = _nextId;
        ++_nextId;
        OutputArchiveHandle handle(currentId, obj->getPersistenceName(), obj->getPythonModule(), self);
        obj->write(handle);
        return currentId;
    }

    int put(std::shared_ptr<Persistable const> obj, std::shared_ptr<Impl> const &self, bool permissive) {
        if (!obj) return 0;
        if (permissive && !obj->isPersistable()) return 0;
        MapItem item(obj, _nextId);
        std::pair<Map::iterator, bool> r = _map.insert(item);
        if (r.second) {
            // We've never seen this object before.  Save it.
            return put(obj.get(), self, permissive);
        } else {
            // We had already saved this object, and insert returned an iterator
            // to the ID we used before; return that.
            return r.first->second;
        }
    }

    void writeFits(fits::Fits &fitsfile) {
        _index.getTable()->getMetadata()->set("AR_NCAT", int(_catalogs.size() + 1),
                                              "# of catalogs in this archive, including the index");
        _index.writeFits(fitsfile);
        int n = 1;
        for (CatalogVector::const_iterator iter = _catalogs.begin(); iter != _catalogs.end(); ++iter, ++n) {
            iter->writeFits(fitsfile);
        }
    }

    Impl() :  _map(), _index(ArchiveIndexSchema::get().schema) {
        std::shared_ptr<daf::base::PropertyList> metadata(new daf::base::PropertyList());
        metadata->set("EXTTYPE", "ARCHIVE_INDEX");
        metadata->set("EXTNAME", "ARCHIVE_INDEX");
        metadata->set("AR_CATN", 0, "# of this catalog relative to the start of this archive");
        _index.getTable()->setMetadata(metadata);
    }

    int _nextId{1};
    Map _map;
    BaseCatalog _index;
    CatalogVector _catalogs;
};

// ----- OutputArchive --------------------------------------------------------------------------------------

OutputArchive::OutputArchive() : _impl(new Impl()) {}

OutputArchive::OutputArchive(OutputArchive const &other)  = default;
// Delegate to copy constructor for backward compatibility
OutputArchive::OutputArchive(OutputArchive &&other) : OutputArchive(other) {}

OutputArchive &OutputArchive::operator=(OutputArchive const &other) = default;
// Delegate to copy assignment for backward compatibility
OutputArchive &OutputArchive::operator=(OutputArchive &&other) { return *this = other; }

OutputArchive::~OutputArchive() = default;

int OutputArchive::put(Persistable const *obj, bool permissive) {
    if (_impl.use_count() != 1) {  // copy on write
        std::shared_ptr<Impl> tmp(new Impl(*_impl));
        _impl.swap(tmp);
    }
    return _impl->put(obj, _impl, permissive);
}

int OutputArchive::put(std::shared_ptr<Persistable const> obj, bool permissive) {
    if (_impl.use_count() != 1) {  // copy on write
        std::shared_ptr<Impl> tmp(new Impl(*_impl));
        _impl.swap(tmp);
    }
    return _impl->put(std::move(obj), _impl, permissive);
}

BaseCatalog const &OutputArchive::getIndexCatalog() const { return _impl->_index; }

BaseCatalog const &OutputArchive::getCatalog(int n) const {
    if (n == 0) return _impl->_index;
    if (std::size_t(n) > _impl->_catalogs.size() || n < 0) {
        throw LSST_EXCEPT(
                pex::exceptions::LengthError,
                (boost::format("Catalog number %d is out of range [0,%d]") % n % _impl->_catalogs.size())
                        .str());
    }
    return _impl->_catalogs[n - 1];
}

std::size_t OutputArchive::countCatalogs() const { return _impl->_catalogs.size() + 1; }

void OutputArchive::writeFits(fits::Fits &fitsfile) const { _impl->writeFits(fitsfile); }

// ----- OutputArchiveHandle ------------------------------------------------------------------------------

BaseCatalog OutputArchiveHandle::makeCatalog(Schema const &schema) { return _impl->makeCatalog(schema); }

void OutputArchiveHandle::saveEmpty() { _impl->saveEmpty(_id, _name, _module); }

void OutputArchiveHandle::saveCatalog(BaseCatalog const &catalog) {
    _impl->saveCatalog(catalog, _id, _name, _module, _catPersistable);
    ++_catPersistable;
}

int OutputArchiveHandle::put(Persistable const *obj, bool permissive) {
    // Handle doesn't worry about copy-on-write, because Handles should only exist
    // while an OutputArchive::put() call is active.
    return _impl->put(obj, _impl, permissive);
}

int OutputArchiveHandle::put(std::shared_ptr<Persistable const> obj, bool permissive) {
    // Handle doesn't worry about copy-on-write, because Handles should only exist
    // while an OutputArchive::put() call is active.
    return _impl->put(std::move(obj), _impl, permissive);
}

OutputArchiveHandle::OutputArchiveHandle(int id, std::string const &name, std::string const &module,
                                         std::shared_ptr<OutputArchive::Impl> impl)
        : _id(id), _catPersistable(0), _name(name), _module(module), _impl(impl) {}

OutputArchiveHandle::~OutputArchiveHandle() = default;
}  // namespace io
}  // namespace table
}  // namespace afw
}  // namespace lsst
