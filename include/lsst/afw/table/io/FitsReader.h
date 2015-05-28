// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_FitsReader_h_INCLUDED
#define AFW_TABLE_IO_FitsReader_h_INCLUDED

#include "lsst/afw/fits.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/FitsSchemaInputMapper.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"

namespace lsst { namespace afw { namespace table { namespace io {

/**
 *  @brief A utility class for reading FITS binary tables.
 *
 *  FitsReader itself provides the implementation for reading standard FITS binary tables
 *  (with a limited subset of FITS column types), but it also allows subclasses to be used
 *  instead, depending on what's actually in the FITS file.  If the FITS header has the key
 *  "AFW_TABLE" with a value other than "BASE", FitsReader::apply consults a registry
 *  of subclasses to retreive one corresponding to that key.  This means the type of
 *  records/tables loaded correctly depends on the file itself, rather than the caller.
 *  For instance, if you load a FITS table corresponding to a saved SourceCatalog using
 *  BaseCatalog::readFits, you'll actually get a BaseCatalog whose record are actually
 *  SourceRecords and whose table is actually a SourceTable.  On the other hand, if you
 *  try to load a non-Source FITS table into a SourceCatalog, you'll get an exception
 *  when it tries to dynamic_cast the table to a SourceTable.
 */
class FitsReader {
public:

    /**
     *  Construct a FitsReader, registering it to be used for all persisted tables with the given tag.
     *
     *  Because they need to live in the static registry, each distinct subclass of FitsReader
     *  should be constructed only once, in a static-scope variable.  The FitsReader constructor
     *  will add a pointer to that variable to the registry.
     */
    explicit FitsReader(std::string const & persistedClassName);

    /**
     *  Create a new Catalog by reading a FITS binary table.
     *
     *  This is the lower-level implementation delegated to by all Catalog::readFits() methods.
     *  It creates a new Catalog of type ContainerT, creates a FitsReader according to the tag
     *  found in the file, then reads the schema and adds records to the Catalog.
     *
     *  @param[in]  fits     An afw::fits::Fits helper that points to a FITS binary table HDU.
     *  @param[in]  ioFlags  A set of subclass-dependent bitflags that control optional aspects of FITS
     *                       persistence.  For instance, SourceFitsFlags are used by SourceCatalog
     *                       to control how to read and write Footprints.
     *  @param[in]  archive  An archive of Persistables containing objects that may be associated
     *                       with table records.  For record subclasses that have associated Persistables
     *                       (e.g. SourceRecord Footprints, or ExposureRecord Psfs), this archive is usually
     *                       persisted in additional HDUs in the FITS file after the main binary table,
     *                       and will be loaded automatically if the passed archive is null.  The explicit
     *                       archive argument is provided only for cases in which the catalog itself is
     *                       part of a larger object, and does not "own" its own archive (e.g. CoaddPsf
     *                       persistence).
     */
    template <typename ContainerT>
    static ContainerT apply(afw::fits::Fits & fits, int ioFlags, PTR(InputArchive) archive=PTR(InputArchive)()) {
        PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
        fits.readMetadata(*metadata, true);
        FitsReader const * reader = _lookupFitsReader(*metadata);
        FitsSchemaInputMapper mapper(*metadata, true);
        reader->_setupArchive(fits, mapper, archive, ioFlags);
        PTR(BaseTable) table = reader->makeTable(mapper, metadata, ioFlags, true);
        ContainerT container(boost::dynamic_pointer_cast<typename ContainerT::Table>(table));
        if (!container.getTable()) {
            throw LSST_EXCEPT(
                pex::exceptions::RuntimeError,
                "Invalid table class for catalog."
            );
        }
        std::size_t nRows = fits.countRows();
        container.reserve(nRows);
        for (std::size_t row = 0; row < nRows; ++row) {
            mapper.readRecord(
                // We need to be able to support reading Catalog<T const>, since it shares the same template
                // as Catalog<T> (which invokes this method in readFits).
                const_cast<typename boost::remove_const<typename ContainerT::Record>::type&>(
                    *container.addNew()
                ),
                fits, row
            );
        }
        return container;
    }

    /**
     *  Create a new Catalog by reading a FITS file.
     *
     *  This is a simply a convenience function that creates an afw::fits::Fits object from either
     *  a string filename or a afw::fits::MemFileManager, then calls the other apply() overload.
     */
    template <typename ContainerT, typename SourceT>
    static ContainerT apply(SourceT & source, int hdu, int ioFlags, PTR(InputArchive) archive=PTR(InputArchive)()) {
        afw::fits::Fits fits(source, "r", afw::fits::Fits::AUTO_CLOSE | afw::fits::Fits::AUTO_CHECK);
        fits.setHdu(hdu);
        return apply<ContainerT>(fits, ioFlags, archive);
    }

    /**
     *  Callback to create a Table object from a FITS binary table schema.
     *
     *  Subclass readers must override to return the appropriate Table subclass.
     *  Most implementations can simply call mapper.finalize() to create the Schema, then construct a
     *  new Table and set its metadata to the given PropertyList.
     *  Readers for record classes that have first-class objects in addition to regular fields
     *  should call mapper.customize() with a custom FitsColumnReader before calling finalize().
     *
     *  @param[in]  mapper    A representation of the FITS binary table schema, capable of producing
     *                        an afw::table::Schema from it while allowing customization of the mapping
     *                        beforehand.
     *  @param[in]  metadata  Entries from the FITS header, which should usually be attached to the
     *                        returned table object via its setMetadata method.
     *  @param[in]  ioFlags   Subclass-dependent bitflags that control optional persistence behaavior
     *                        (see e.g. SourceFitsFlags).
     *  @param[in]  stripMetadata   If True, remove entries from the metadata that were added by the
     *                              persistence code.
     */
    virtual PTR(BaseTable) makeTable(
        FitsSchemaInputMapper & mapper,
        PTR(daf::base::PropertyList) metadata,
        int ioFlags,
        bool stripMetadata
    ) const;

    /**
     *  Callback that should return true if the FitsReader subclass makes use of an InputArchive to read
     *  first-class objects from additional FITS HDUs.
     */
    virtual bool usesArchive(int ioFlags) const { return false; }

    virtual ~FitsReader() {}

private:

    static FitsReader const * _lookupFitsReader(daf::base::PropertyList const & metadata);

    void _setupArchive(
        afw::fits::Fits & fits,
        FitsSchemaInputMapper & mapper,
        PTR(InputArchive) archive,
        int ioFlags
    ) const;

};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_FitsReader_h_INCLUDED
