// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_OutputArchive_h_INCLUDED
#define AFW_TABLE_IO_OutputArchive_h_INCLUDED

#include "boost/noncopyable.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/Catalog.h"

namespace lsst { namespace afw { 

namespace fits {

class Fits;

} // namespace fits

namespace table { namespace io {

class Persistable;

/**
 *  @brief A multi-catalog archive object used to save table::io::Persistable objects.
 *
 *  OutputArchive should generally be used directly only by objects that do not themselves
 *  inherit from Persistable, but contain many objects that do (such as Exposure).  It provides
 *  an interface for adding objects to the archive (put()), transforming them into catalogs
 *  that can be retrieved directly or written to a FITS file.  The first catalog is an index
 *  that indicates which rows of the subsequent catalogs correspond to each object.
 *
 *  See getIndexCatalog() for a more detailed description of the index.
 */
class OutputArchive {
public:

    class Handle;
    class CatalogProxy;

    /// Construct an empty OutputArchive containing no objects.
    OutputArchive();

    /// Copy-construct an OutputArchive.  Saved objects are not deep-copied.
    OutputArchive(OutputArchive const & other);

    /// Assign from another OutputArchive.  Saved objects are not deep-copied.
    OutputArchive & operator=(OutputArchive const & other);

    // (trivial) destructor must be defined in the source for pimpl idiom.
    ~OutputArchive();

    /**
     *  @brief Save an object to the archive and return a unique ID that can be used
     *         to retrieve it from an InputArchive.
     *
     *  If the given pointer has already been saved, it will not be written again
     *  and the same ID will be returned as the first time it was saved.
     *
     *  If the given pointer is null, the returned ID is always 0, which may be used
     *  to retrieve null pointers from an InputArchive.
     */
    int put(Persistable const * obj);

    /**
     *  @brief Return the index catalog that specifies where objects are stored in the
     *         data catalogs.
     */
    BaseCatalog const & getIndexCatalog() const;

    /// Return the nth catalog.  Catalog 0 is always the index catalog.
    BaseCatalog const & getCatalog(int n) const;

    /// Return the total number of catalogs, including the index.
    int countCatalogs() const;

    /**
     *  @brief Write the archive to an already-open FITS object.
     *
     *  Always appends new HDUs.
     *
     *  @param[in] fitsfile     Open FITS object to write to.
     */
    void writeFits(fits::Fits & fitsfile) const;

private:

    class Impl;

    PTR(Impl) _impl;
};

/**
 *  @brief An append-only catalog proxy used by implementations of Persistable::write.
 *
 *  A CatalogProxy can only be obtained by calling Handle::addCatalog, and provides
 *  an interface for adding records to a single catalog (and pretty much nothing else).
 */
class OutputArchive::CatalogProxy {
public:

    /// Append a new record to the catalogand return it.
    PTR(BaseRecord) addRecord();

    /// Shallow copy constructor.
    CatalogProxy(CatalogProxy const & other);

    /// Shallow assignment operator.
    CatalogProxy & operator=(CatalogProxy const & other);

    ~CatalogProxy();

private:

    friend class OutputArchive::Handle;

    CatalogProxy(PTR(BaseRecord) index, PTR(Impl) impl);

    PTR(BaseRecord) _index;
    PTR(Impl) _impl;
};

/**
 *  @brief An object passed to Persistable::write to allow it to persist itself.
 *
 *  Handle provides an interface to add additional catalogs and save nested
 *  Persistables to the same archive.
 */
class OutputArchive::Handle : private boost::noncopyable {
public:

    /**
     *  @brief Start a new catalog with the given schema.
     *
     *  The returned object may actually point to a new catalog in the archive,
     *  or it may just append records to an existing catalog in the archive that
     *  has the same schema.
     */
    CatalogProxy addCatalog(Schema const & schema);

    /**
     *  @brief Save a nested Persistable to the same archive.
     *
     *  @copydoc OutputArchive::put.
     */
    int put(Persistable const * obj);

    ~Handle();

private:

    friend class OutputArchive::Impl;

    Handle(int id, std::string const & name, PTR(Impl) impl);

    int _id;
    int _catPersistable;
    std::string _name;
    PTR(Impl) _impl;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_OutputArchive_h_INCLUDED
