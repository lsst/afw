// -*- lsst-c++ -*-

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/ArchiveIndexSchema.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

ArchiveIndexSchema const & indexKeys = ArchiveIndexSchema::get();

// Functor to sort records by ID and then by catPersistable
struct IndexSortCompare {
    bool operator()(BaseRecord const & a, BaseRecord const & b) const {
        if (a.get(indexKeys.id) < b.get(indexKeys.id)) {
            return true;
        }
        if (a.get(indexKeys.id) == b.get(indexKeys.id)) {
            return a.get(indexKeys.catPersistable) < b.get(indexKeys.catPersistable);
        }
        return false;
    }
};

} // anonymous

// ----- InputArchive::Impl ---------------------------------------------------------------------------------

class InputArchive::Impl : private boost::noncopyable {
public:

    PTR(Persistable) get(int id, InputArchive const & self) {
        PTR(Persistable) empty;
        if (id == 0) return empty;
        std::pair<Map::iterator,bool> r = _map.insert(std::make_pair(id, empty));
        if (r.second) {
            // insertion successful means we haven't reassembled this object yet; do that now.
            CatalogVector factoryArgs;
            // iterate over records in index with this ID; we know they're sorted by ID and then
            // by catPersistable, so we can just append to factoryArgs.
            std::string name;
            std::string module;
            for (
                BaseCatalog::iterator indexIter = _index.find(id, indexKeys.id);
                indexIter != _index.end() && indexIter->get(indexKeys.id) == id; 
                ++indexIter
            ) {
                if (name.empty()) {
                    name = indexIter->get(indexKeys.name);
                } else if (name != indexIter->get(indexKeys.name)) {
                    throw LSST_EXCEPT(
                        MalformedArchiveError,
                        (boost::format(
                            "Inconsistent name in index for ID %d; got '%s', expected '%s'"
                        ) % indexIter->get(indexKeys.id) % indexIter->get(indexKeys.name) % name).str()
                    );
                }
                if (module.empty()) {
                    module = indexIter->get(indexKeys.module);
                } else if (module != indexIter->get(indexKeys.module)) {
                    throw LSST_EXCEPT(
                        MalformedArchiveError,
                        (boost::format(
                            "Inconsistent module in index for ID %d; got '%s', expected '%s'"
                        ) % indexIter->get(indexKeys.id) % indexIter->get(indexKeys.module) % module).str()
                    );
                }
                std::size_t catN = indexIter->get(indexKeys.catArchive)-1;
                if (catN >= _catalogs.size()) {
                    throw LSST_EXCEPT(
                        MalformedArchiveError,
                        (boost::format(
                            "Invalid catalog number in index for ID %d; got '%d', max is '%d'"
                        ) % indexIter->get(indexKeys.id) % catN % _catalogs.size()).str()
                    );
                }
                BaseCatalog & fullCatalog = _catalogs[catN];
                std::size_t i1 = indexIter->get(indexKeys.row0);
                std::size_t i2 = i1 + indexIter->get(indexKeys.nRows);
                if (i2 > fullCatalog.size()) {
                    throw LSST_EXCEPT(
                        MalformedArchiveError,
                        (boost::format(
                            "Index and data catalogs do not agree for ID %d; catalog %d has %d rows, not %d"
                        ) % indexIter->get(indexKeys.id)
                         % indexIter->get(indexKeys.catArchive) % fullCatalog.size() % i2).str()
                    );
                }
                factoryArgs.push_back(
                    BaseCatalog(fullCatalog.getTable(), fullCatalog.begin() + i1, fullCatalog.begin() + i2)
                );
            }
            try {
                PersistableFactory const & factory = PersistableFactory::lookup(name, module);
                r.first->second = factory.read(self, factoryArgs);
            } catch (pex::exceptions::Exception & err) {
                LSST_EXCEPT_ADD(
                    err, (boost::format("loading object with id=%d, name='%s'") % id % name).str()
                );
                throw;
            }
        }
        assert(r.first->second);
        return r.first->second;
    }

    Map const & getAll(InputArchive const & self) {
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

    Impl(BaseCatalog const & index, CatalogVector const & catalogs=CatalogVector()) :
        _index(index), _catalogs(catalogs)
    {
        if (index.getSchema() != indexKeys.schema) {
            throw LSST_EXCEPT(
                pex::exceptions::RuntimeErrorException,
                "Incorrect schema for index catalog"
            );
        }
        _map.insert(std::make_pair(0, PTR(Persistable)()));
        _index.sort(IndexSortCompare());
    }

    Map _map;
    BaseCatalog _index;
    CatalogVector _catalogs;
};

// ----- InputArchive ---------------------------------------------------------------------------------------

InputArchive::InputArchive() : _impl(new Impl()) {}

InputArchive::InputArchive(PTR(Impl) impl) : _impl(impl) {}

InputArchive::InputArchive(BaseCatalog const & index, CatalogVector const & catalogs) :
    _impl(new Impl(index, catalogs))
{}

InputArchive::InputArchive(InputArchive const & other) : _impl(other._impl) {}

InputArchive & InputArchive::operator=(InputArchive const & other) {
    _impl = other._impl;
    return *this;
}

InputArchive::~InputArchive() {}

PTR(Persistable) InputArchive::get(int id) const { return _impl->get(id, *this); }

InputArchive::Map const & InputArchive::getAll() const { return _impl->getAll(*this); }

InputArchive InputArchive::readFits(fits::Fits & fitsfile) {
    BaseCatalog index = BaseCatalog::readFits(fitsfile);
    PTR(daf::base::PropertyList) metadata = index.getTable()->popMetadata();
    assert(metadata); // BaseCatalog::readFits should always read metadata, even if there's nothing there
    if (metadata->get<std::string>("EXTTYPE") != "ARCHIVE_INDEX") {
        throw LSST_FITS_EXCEPT(
            fits::FitsError,
            fitsfile,
            boost::format("Wrong value for archive index EXTTYPE: '%s'")
            % metadata->get<std::string>("EXTTYPE")
        );
    }
    int nCatalogs = metadata->get<int>("AR_NCAT");
    // The archive isn't fully constructed, but we need something to pass to BaseCatalog::readFits()
    // The order in which we've saved things should ensure that as of the time we read a catalog,
    // all the Persistables it needs to get from the archive are already available.
    PTR(Impl) impl(new Impl(index));
    PTR(InputArchive) self(new InputArchive(impl));
    impl->_catalogs.reserve(nCatalogs);
    for (int n = 1; n < nCatalogs; ++n) {
        fitsfile.setHdu(1, true); // increment HDU by one
        impl->_catalogs.push_back(BaseCatalog::readFits(fitsfile, self));
        metadata = impl->_catalogs.back().getTable()->popMetadata();
        if (metadata->get<std::string>("EXTTYPE") != "ARCHIVE_DATA") {
            throw LSST_FITS_EXCEPT(
                fits::FitsError,
                fitsfile,
                boost::format("Wrong value for archive data EXTTYPE: '%s'")
                % metadata->get<std::string>("EXTTYPE")
            );
        }
        if (metadata->get<int>("AR_CATN") != n) {
            throw LSST_FITS_EXCEPT(
                fits::FitsError,
                fitsfile,
                boost::format("Incorrect order for archive catalogs: AR_CATN=%d found at position %d")
                % metadata->get<int>("AR_CATN") % n
            );
        }
    }
    return InputArchive(impl);
}

}}}} // namespace lsst::afw::table::io
