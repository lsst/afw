// -*- lsst-c++ -*-

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/ArchiveIndexSchema.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/fits.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

namespace {

ArchiveIndexSchema const& indexKeys = ArchiveIndexSchema::get();

// Functor to sort records by ID and then by catPersistable
struct IndexSortCompare {
    bool operator()(BaseRecord const& a, BaseRecord const& b) const {
        if (a.get(indexKeys.id) < b.get(indexKeys.id)) {
            return true;
        }
        if (a.get(indexKeys.id) == b.get(indexKeys.id)) {
            return a.get(indexKeys.catPersistable) < b.get(indexKeys.catPersistable);
        }
        return false;
    }
};

}  // anonymous

// ----- InputArchive::Impl ---------------------------------------------------------------------------------

class InputArchive::Impl {
public:
    std::shared_ptr<Persistable> get(int id, InputArchive const& self) {
        std::shared_ptr<Persistable> empty;
        if (id == 0) return empty;
        std::pair<Map::iterator, bool> r = _map.insert(std::make_pair(id, empty));
        if (r.second) {
            // insertion successful means we haven't reassembled this object yet; do that now.
            CatalogVector factoryArgs;
            // iterate over records in index with this ID; we know they're sorted by ID and then
            // by catPersistable, so we can just append to factoryArgs.
            std::string name;
            std::string module;
            for (BaseCatalog::iterator indexIter = _index.find(id, indexKeys.id);
                 indexIter != _index.end() && indexIter->get(indexKeys.id) == id; ++indexIter) {
                if (name.empty()) {
                    name = indexIter->get(indexKeys.name);
                } else if (name != indexIter->get(indexKeys.name)) {
                    throw LSST_EXCEPT(
                            MalformedArchiveError,
                            (boost::format("Inconsistent name in index for ID %d; got '%s', expected '%s'") %
                             indexIter->get(indexKeys.id) % indexIter->get(indexKeys.name) %
                             name).str());
                }
                if (module.empty()) {
                    module = indexIter->get(indexKeys.module);
                } else if (module != indexIter->get(indexKeys.module)) {
                    throw LSST_EXCEPT(
                            MalformedArchiveError,
                            (boost::format(
                                     "Inconsistent module in index for ID %d; got '%s', expected '%s'") %
                             indexIter->get(indexKeys.id) % indexIter->get(indexKeys.module) %
                             module).str());
                }
                int catArchive = indexIter->get(indexKeys.catArchive);
                if (catArchive == ArchiveIndexSchema::NO_CATALOGS_SAVED) {
                    break;  // object was written with saveEmpty, and hence no catalogs.
                }
                std::size_t catN = catArchive - 1;
                if (catN >= _catalogs.size()) {
                    throw LSST_EXCEPT(
                            MalformedArchiveError,
                            (boost::format(
                                     "Invalid catalog number in index for ID %d; got '%d', max is '%d'") %
                             indexIter->get(indexKeys.id) % catN % _catalogs.size())
                                    .str());
                }
                BaseCatalog& fullCatalog = _catalogs[catN];
                std::size_t i1 = indexIter->get(indexKeys.row0);
                std::size_t i2 = i1 + indexIter->get(indexKeys.nRows);
                if (i2 > fullCatalog.size()) {
                    throw LSST_EXCEPT(MalformedArchiveError,
                                      (boost::format("Index and data catalogs do not agree for ID %d; "
                                                     "catalog %d has %d rows, not %d") %
                                       indexIter->get(indexKeys.id) % indexIter->get(indexKeys.catArchive) %
                                       fullCatalog.size() %
                                       i2).str());
                }
                factoryArgs.push_back(BaseCatalog(fullCatalog.getTable(), fullCatalog.begin() + i1,
                                                  fullCatalog.begin() + i2));
            }
            try {
                PersistableFactory const& factory = PersistableFactory::lookup(name, module);
                r.first->second = factory.read(self, factoryArgs);
            } catch (pex::exceptions::Exception& err) {
                LSST_EXCEPT_ADD(err,
                                (boost::format("loading object with id=%d, name='%s'") % id % name).str());
                throw;
            }
            // If we're loading the object for the first time, and we've failed, we should have already
            // thrown an exception, and we assert that here.
            assert(r.first->second);
        } else if (!r.first->second) {
            // If we'd already tried and failed to load this object before - but we'd caught the exception
            // previously (because the calling code didn't consider that to be a fatal error) - we'll
            // just throw an exception again.  While we can't know exactly what was thrown before,
            // it's most likely it was a NotFoundError because a needed extension package was not setup.
            // And conveniently it's appropriate to throw that here too, since now the problem is that
            // the object should have been loaded into the cache and it wasn't found there.
            throw LSST_EXCEPT(pex::exceptions::NotFoundError,
                              (boost::format("Not trying to reload object with id=%d; a previous attempt to "
                                             "load it already failed.") %
                               id).str());
        }
        return r.first->second;
    }

    Map const& getAll(InputArchive const& self) {
        int id = 0;
        for (BaseCatalog::iterator indexIter = _index.begin(); indexIter != _index.end(); ++indexIter) {
            if (indexIter->get(indexKeys.id) != id) {
                id = indexIter->get(indexKeys.id);
                get(id, self);
            }
        }
        return _map;
    }

    Impl() : _index(ArchiveIndexSchema::get().schema) {}

    Impl(BaseCatalog const& index, CatalogVector const& catalogs) : _index(index), _catalogs(catalogs) {
        if (index.getSchema() != indexKeys.schema) {
            throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Incorrect schema for index catalog");
        }
        _map.insert(std::make_pair(0, std::shared_ptr<Persistable>()));
        _index.sort(IndexSortCompare());
    }

    // No copying
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // No moving
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    Map _map;
    BaseCatalog _index;
    CatalogVector _catalogs;
};

// ----- InputArchive ---------------------------------------------------------------------------------------

InputArchive::InputArchive() : _impl(new Impl()) {}

InputArchive::InputArchive(std::shared_ptr<Impl> impl) : _impl(impl) {}

InputArchive::InputArchive(BaseCatalog const& index, CatalogVector const& catalogs)
        : _impl(new Impl(index, catalogs)) {}

InputArchive::InputArchive(InputArchive const& other) : _impl(other._impl) {}
// Delegate to copy constructor for backwards compatibility
InputArchive::InputArchive(InputArchive && other) : InputArchive(other) {}

InputArchive& InputArchive::operator=(InputArchive const& other) {
    _impl = other._impl;
    return *this;
}
// Delegate to copy assignment for backwards compatibility
InputArchive& InputArchive::operator=(InputArchive && other) { return *this = other; }

InputArchive::~InputArchive() = default;

std::shared_ptr<Persistable> InputArchive::get(int id) const { return _impl->get(id, *this); }

InputArchive::Map const& InputArchive::getAll() const { return _impl->getAll(*this); }

InputArchive InputArchive::readFits(fits::Fits& fitsfile) {
    BaseCatalog index = BaseCatalog::readFits(fitsfile);
    std::shared_ptr<daf::base::PropertyList> metadata = index.getTable()->popMetadata();
    assert(metadata);  // BaseCatalog::readFits should always read metadata, even if there's nothing there
    if (metadata->get<std::string>("EXTTYPE") != "ARCHIVE_INDEX") {
        throw LSST_FITS_EXCEPT(fits::FitsError, fitsfile,
                               boost::format("Wrong value for archive index EXTTYPE: '%s'") %
                                       metadata->get<std::string>("EXTTYPE"));
    }
    int nCatalogs = metadata->get<int>("AR_NCAT");
    CatalogVector catalogs;
    catalogs.reserve(nCatalogs);
    for (int n = 1; n < nCatalogs; ++n) {
        fitsfile.setHdu(1, true);  // increment HDU by one
        catalogs.push_back(BaseCatalog::readFits(fitsfile));
        metadata = catalogs.back().getTable()->popMetadata();
        if (metadata->get<std::string>("EXTTYPE") != "ARCHIVE_DATA") {
            throw LSST_FITS_EXCEPT(fits::FitsError, fitsfile,
                                   boost::format("Wrong value for archive data EXTTYPE: '%s'") %
                                           metadata->get<std::string>("EXTTYPE"));
        }
        if (metadata->get<int>("AR_CATN") != n) {
            throw LSST_FITS_EXCEPT(
                    fits::FitsError, fitsfile,
                    boost::format("Incorrect order for archive catalogs: AR_CATN=%d found at position %d") %
                            metadata->get<int>("AR_CATN") % n);
        }
    }
    std::shared_ptr<Impl> impl(new Impl(index, catalogs));
    return InputArchive(impl);
}
}
}
}
}  // namespace lsst::afw::table::io
