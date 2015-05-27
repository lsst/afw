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
 *  @brief A Reader subclass for FITS binary tables.
 *
 *  FitsReader itself provides the implementation for reading standard FITS binary tables
 *  (with a limited subset of FITS column types), but it also allows subclasses to be used
 *  instead, depending on what's actually in the FITS file.  If the FITS header has the key
 *  "AFW_TABLE" with a value other than "BASE", FitsReader::read consults a registry
 *  of subclasses to retreive one corresponding to that key.  This means the type of
 *  records/tables loaded correctly depends on the file itself, rather than the caller.
 *  For instance, if you load a FITS table corresponding to a saved SourceCatalog using
 *  BaseCatalog::readFits, you'll actually get a BaseCatalog whose record are actually
 *  SourceRecords and whose table is actually a SourceTable.  On the other hand, if you
 *  try to load a non-Source FITS table into a SourceCatalog, you'll get an exception
 *  when it tries to dynamic_cast the table to a SourceTable.
 *
 *  Because they need to live in the static registry, each distinct subclass of FitsReader
 *  should be constructed only once, in a static-scope variable.  The FitsReader constructor
 *  will add a pointer to that variable to the registry.
 */
class FitsReader {
public:

    explicit FitsReader(std::string const & persistedClassName);

    template <typename ContainerT>
    static ContainerT apply(afw::fits::Fits & fits, int ioFlags, PTR(InputArchive) archive=nullptr) {
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

    template <typename ContainerT, typename SourceT>
    static ContainerT apply(SourceT & source, int hdu, int ioFlags, PTR(InputArchive) archive=nullptr) {
        afw::fits::Fits fits(source, "r", afw::fits::Fits::AUTO_CLOSE | afw::fits::Fits::AUTO_CHECK);
        fits.setHdu(hdu);
        return apply<ContainerT>(fits, ioFlags, archive);
    }

    virtual PTR(BaseTable) makeTable(
        FitsSchemaInputMapper & mapper,
        PTR(daf::base::PropertyList) metadata,
        int ioFlags,
        bool stripMetadata
    ) const;

    virtual bool usesArchive(int ioFlags) const { return false; }

    virtual ~FitsReader() {}

private:

    friend class Schemaa;

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
