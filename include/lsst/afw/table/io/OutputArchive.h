// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_OutputArchive_h_INCLUDED
#define AFW_TABLE_IO_OutputArchive_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst { namespace afw { namespace table {

class Schema;
class BaseRecord;
template <typename RecordT> class CatalogT;
typedef CatalogT<BaseRecord> BaseCatalog;

namespace io {

class Persistable;
class OutputArchiveHandle;

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

    friend class OutputArchiveHandle;

    /// Construct an empty OutputArchive containing no objects.
    OutputArchive();

    /// Copy-construct an OutputArchive.  Saved objects are not deep-copied.
    OutputArchive(OutputArchive const & other);

    /// Assign from another OutputArchive.  Saved objects are not deep-copied.
    OutputArchive & operator=(OutputArchive const & other);

    // (trivial) destructor must be defined in the source for pimpl idiom.
    ~OutputArchive();

    //@{
    /**
     *  @brief Save an object to the archive and return a unique ID that can be used
     *         to retrieve it from an InputArchive.
     *
     *  If permissive is true and obj->isPersistable() is false, the object will not
     *  be saved but 0 will be returned instead of throwing an exception.
     *
     *  If the given pointer has already been saved, it will not be written again
     *  and the same ID will be returned as the first time it was saved.
     *
     *  If the given pointer is null, the returned ID is always 0, which may be used
     *  to retrieve null pointers from an InputArchive.
     *
     *  It is expected that the CONST_PTR form will usually be used, as Persistables
     *  are typically held by PTR.  But we expose the lower-level raw-pointer form
     *  so users aren't forced to clone objects before persisting them if they don't
     *  already have a PTR.
     *
     *  The implementation of 'put' does not provide any exception safety; if the object
     *  being saved (or any nested object) throws an exception, the entire archive may
     *  be in an inconsistent state and should not be saved.
     */
    int put(Persistable const * obj, bool permissive=false);
    int put(CONST_PTR(Persistable) obj, bool permissive=false) { return put(obj.get(), permissive); }
    //@}

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
 *  @brief An object passed to Persistable::write to allow it to persist itself.
 *
 *  OutputArchiveHandle provides an interface to add additional catalogs and save nested
 *  Persistables to the same archive.
 */
class OutputArchiveHandle {
public:

    /**
     *  @brief Return a new, empty catalog with the given schema.
     *
     *  All catalogs passed to saveCatalog should be originally
     *  created by makeCatalog, or at least share the same table.
     */
    BaseCatalog makeCatalog(Schema const & schema);

    /**
     *  @brief Save a catalog in the archive.
     *
     *  The catalog must have been created using makeCatalog,
     *  or be a shallow copy or subset of such a catalog.
     */
    void saveCatalog(BaseCatalog const & catalog);

    //@{
    /**
     *  @brief Save a nested Persistable to the same archive.
     *
     *  @copydoc OutputArchive::put.
     */
    int put(Persistable const * obj, bool permissive=false);
    int put(CONST_PTR(Persistable) obj, bool permissive=false) { return put(obj.get(), permissive); }
    //@}

    ~OutputArchiveHandle();

    // No copying
    OutputArchiveHandle (const OutputArchiveHandle&) = delete;
    OutputArchiveHandle& operator=(const OutputArchiveHandle&) = delete;

    // No moving
    OutputArchiveHandle (OutputArchiveHandle&&) = delete;
    OutputArchiveHandle& operator=(OutputArchiveHandle&&) = delete;

private:

    friend class OutputArchive::Impl;

    OutputArchiveHandle(
        int id, std::string const & name, std::string const & module,
        PTR(OutputArchive::Impl) impl
    );

    int _id;
    int _catPersistable;
    std::string _name;
    std::string _module;
    PTR(OutputArchive::Impl) _impl;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_OutputArchive_h_INCLUDED
