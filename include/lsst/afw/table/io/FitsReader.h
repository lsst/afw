// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_FitsReader_h_INCLUDED
#define AFW_TABLE_IO_FitsReader_h_INCLUDED

#include "lsst/afw/fits.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/io/Reader.h"
#include "lsst/afw/table/io/InputArchive.h"

namespace lsst { namespace afw { namespace table { namespace io {

/**
 *  @brief A Reader subclass for FITS binary tables.
 *
 *  FitsReader itself provides the implementation for reading standard FITS binary tables
 *  (with a limited subset of FITS column types), but it also allows subclasses to be used
 *  instead, depending on what's actually in the FITS file.  If the FITS header has the key
 *  "AFW_TABLE" with a value other than "BASE", FitsReader::make consults a registry of
 *  and constructs the subclass corresponding to that key.  This means the type of
 *  records/tables loaded correctly depends on the file itself, rather than the caller.
 *  For instance, if you load a FITS table corresponding to a saved SourceCatalog using
 *  BaseCatalog::readFits, you'll actually get a BaseCatalog whose record are actually
 *  SourceRecords and whose table is actually a SourceTable.  On the other hand, if you
 *  try to load a non-Source FITS table into a SourceCatalog, you'll get an exception
 *  when it tries to dynamic_cast the table to a SourceTable.
 */
class FitsReader : public Reader {
public:

    typedef afw::fits::Fits Fits;
    
    /**
     *  @brief Factory class used to construct FitsReaders.
     *
     *  The constructor for Factory puts a raw pointer to itself in a global registry.
     *  This means Factory and its subclasses should only be constructed as namespace-scope
     *  objects (so they never go out of scope, and automatically get registered).
     *
     *  Subclasses should use this via its derived template class FactoryT.
     */
    class Factory {
    public:

        /// Create a new FITS reader from a cfitsio pointer holder and (optional) input archive.
        virtual PTR(FitsReader) operator()(Fits * fits, PTR(InputArchive) archive) const = 0;

        virtual ~Factory() {}

        /// Create a factory that will be used when the AFW_TYPE fits key matches the given name.
        explicit Factory(std::string const & name);

    };

    /**
     *  @brief Subclass for Factory that constructs a FitsReader.
     *
     *  Subclasses should use this by providing a the appropriate constructor and then declaring
     *  a static data member or namespace-scope FactoryT instance templated over the subclass type.
     *  This will register the subclass so it can be used with FitsReader::make.
     */
    template <typename ReaderT>
    class FactoryT : public Factory {
    public:

        /// Create a new FITS reader from a cfitsio pointer holder and (optional) input archive.
        virtual PTR(FitsReader) operator()(Fits * fits, PTR(InputArchive) archive) const {
            return boost::make_shared<ReaderT>(fits, archive);
        }

        /// Create a factory that will be used when the AFW_TYPE fits key matches the given name.
        explicit FactoryT(std::string const & name) : Factory(name) {}

    };

    /**
     *  @brief Look for the header key (AFW_TYPE) that tells us the type of the FitsReader to use,
     *         then make it using the registered factory.
     */
    static PTR(FitsReader) make(Fits * fits, PTR(io::InputArchive) archive);

    /**
     *  @brief Entry point for reading FITS files into arbitrary containers.
     *
     *  This does the work of opening the file, calling FitsReader::make, and then calling
     *  Reader::read.
     */
    template <typename ContainerT, typename SourceT>
    static ContainerT apply(SourceT & source, int hdu) {
        Fits fits(source, "r", Fits::AUTO_CLOSE | Fits::AUTO_CHECK);
        fits.setHdu(hdu);
        return apply<ContainerT>(fits);
    }

    /// @brief Low-level entry point for reading FITS files into arbitrary containers.
    template <typename ContainerT>
    static ContainerT apply(Fits & fits, PTR(io::InputArchive) archive = PTR(io::InputArchive)()) {
        PTR(FitsReader) reader = make(&fits, archive);
        return reader->template read<ContainerT>();
    }

    /**
     *  @brief Construct from a wrapped cfitsio pointer and (ignored) InputArchive.
     *
     *  Subclasses that require an InputArchive should accept the one that is passed in,
     *  but may need to construct their own from the HDUs following the catalog HDU(s)
     *  if this pointer is null.
     */
    explicit FitsReader(Fits * fits, PTR(InputArchive)) : _fits(fits) {}

protected:

    /// @copydoc Reader::_readTable
    virtual PTR(BaseTable) _readTable();

    /// @copydoc Reader::_readRecord
    virtual PTR(BaseRecord) _readRecord(PTR(BaseTable) const & table);

    /// @brief Should be called by any reimplementation of _readTable.
    void _startRecords(BaseTable & table);

    struct ProcessRecords;

    Fits * _fits;         // cfitsio pointer in a conveniencer wrapper
    std::size_t _row;     // which row we're currently reading
private:

    friend class afw::table::Schema;

    // Implementation for Schema's constructors that take PropertyLists;
    // it's here to keep FITS-related code a little more centralized.
    static void _readSchema(
        Schema & schema,
        daf::base::PropertyList & metadata,
        bool stripMetadata
    );

    std::size_t _nRows;   // how many total records there are in the FITS table
    boost::shared_ptr<ProcessRecords> _processor; // a private Schema::forEach functor that reads records
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_FitsReader_h_INCLUDED
