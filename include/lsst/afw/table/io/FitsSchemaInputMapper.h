// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_FitsSchemaInputMapper_h_INCLUDED
#define AFW_TABLE_IO_FitsSchemaInputMapper_h_INCLUDED

#include "lsst/afw/fits.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseColumnView.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

/**
 *  Polymorphic reader interface used to read different kinds of objects from
 *  one or more FITS binary table columns.
 */
class FitsColumnReader {
public:
    FitsColumnReader() noexcept = default;

    // Neither copyable nor moveable.
    FitsColumnReader(FitsColumnReader const &) = delete;
    FitsColumnReader(FitsColumnReader &&) = delete;
    FitsColumnReader &operator=(FitsColumnReader const &) = delete;
    FitsColumnReader &operator=(FitsColumnReader &&) = delete;

    /**
     *  Optionally read ahead and cache values from multiple rows.
     *
     *  Subclasses are not required to implement this method; if they do, they
     *  should indicate to readCell that cached values should be used instead.
     *  Subclasses should not assume that prepRead will always be called,
     *  however.
     *
     *  @param[in] firstRow Index of the first row to read.
     *  @param[in] nRows    Number of rows to read.
     *  @param[in] fits     FITS file manager object.
     */
    virtual void prepRead(std::size_t firstRow, std::size_t nRows, fits::Fits & fits) {}

    /**
     *  Read values from a single row.
     *
     *  @param[in,out] record   Record to populate.
     *  @param[in]     row      Index of the row to read from.
     *  @param[in]     fits     FITS file manager object.
     *  @param[in]     archive  Archive holding persisted objects, loaded from
     *                          other HDUs.  May be null.
     */
    virtual void readCell(BaseRecord &record, std::size_t row, fits::Fits &fits,
                          std::shared_ptr<InputArchive> const &archive) const = 0;

    virtual ~FitsColumnReader() noexcept = default;
};

/**
 *  A structure that describes a field as a collection of related strings read from the FITS header.
 */
struct FitsSchemaItem {
    int column;         // column number (0-indexed); -1 for Flag fields
    int bit;            // flag bit number (0-indexed); -1 for non-Flag fields
    std::string ttype;  // name of the field (from TTYPE keys)
    std::string tform;  // FITS column format code (from TFORM keys)
    std::string tccls;  // which field class to use (from our own TCCLS keys)
    std::string tunit;  // field units (from TUNIT keys)
    std::string doc;    // field docs (from comments on TTYPE keys)

    explicit FitsSchemaItem(int column_, int bit_) : column(column_), bit(bit_) {}
};

/**
 *  A class that describes a mapping from a FITS binary table to an afw::table Schema.
 *
 *  A FitsSchemaInputMapper is created every time a FITS binary table is read into an afw::table
 *  catalog, allowing limited customization of the mapping between on-disk FITS table columns
 *  an in-memory fields by subclasses of BaseTable.
 *
 *  The object is constructed from a daf::base::PropertyList that represents the FITS header,
 *  which is used to populate a custom container of FitsSchemaItems.  These can then be retrieved
 *  by name or column number via the find() methods, allowing the user to create custom readers
 *  for columns or groups of columns via addColumnReader().  They can also be removed from the
 *  "regular" fields via the erase() method.  Those regular fields are filled in by the finalize()
 *  method, which automatically generates mappings for any FitsSchemaItems that have not been
 *  removed by calls to erase().  Once finalize() has been called, readRecord() may be called
 *  repeatedly to read FITS rows into record objects according to the mapping that has been
 *  defined.
 */
class FitsSchemaInputMapper {
public:
    using Item = FitsSchemaItem;

    /**
     *  When processing each column, divide this number by the record size (in
     *  bytes) and ask CFITSIO to read this many that of values from that
     *  column in a single call.
     *
     *  Both FITS binary tables and afw.table are stored row-major, so reading
     *  multiple rows from a single column at a time leads to nonsequential
     *  reads.  But given the way the I/O code is structured, we tend to get
     *  nonsequential reads anyway, and it seems the per-call overload to
     *  CFITSIO is sufficiently high that it's best to do this anyway for all
     *  but the largest record sizes.
     */
    static std::size_t PREPPED_ROWS_FACTOR;

    /// Construct a mapper from a PropertyList of FITS header values, stripping recognized keys if desired.
    FitsSchemaInputMapper(daf::base::PropertyList &metadata, bool stripMetadata);

    FitsSchemaInputMapper(FitsSchemaInputMapper const &);
    FitsSchemaInputMapper(FitsSchemaInputMapper &&);
    FitsSchemaInputMapper &operator=(FitsSchemaInputMapper const &);
    FitsSchemaInputMapper &operator=(FitsSchemaInputMapper &&);
    ~FitsSchemaInputMapper();

    /**
     *  Set the Archive to an externally-provided one, overriding any that may have been read.
     */
    void setArchive(std::shared_ptr<InputArchive> archive);

    /**
     *  Set the Archive by reading from the HDU specified by the AR_HDU header entry.
     *
     *  Returns true on success, false if there is no AR_HDU entry.
     */
    bool readArchive(afw::fits::Fits &fits);

    /// Return true if the mapper has an InputArchive.
    bool hasArchive() const;

    /**
     *  Find an item with the given column name (ttype), returning nullptr if no such column exists.
     *
     *  The returned pointer is owned by the mapper object, and should not be deleted.  It is
     *  invalidated by calls to either erase() or finalize().
     */
    Item const *find(std::string const &ttype) const;

    /**
     *  Find an item with the given column number, returning nullptr if no such column exists.
     *
     *  The returned pointer is owned by the mapper object, and should not be deleted.  It is
     *  invalidated by calls to either erase() or finalize().
     */
    Item const *find(int column) const;

    /**
     *  Remove the given item (which should have been retrieved via find()) from the mapping, preventing it
     *  from being included in the regular fields added by finalize().
     */
    void erase(Item const *item);

    /**
     *  Remove the item with the given column name (ttype) from the mapping, preventing it
     *  from being included in the regular fields added by finalize().
     */
    void erase(std::string const &ttype);

    /**
     *  Remove the item at the given column position from the mapping, preventing it
     *  from being included in the regular fields added by finalize().
     */
    void erase(int column);

    /**
     *  Customize a mapping by providing a FitsColumnReader instance that will be invoked by readRecords().
     */
    void customize(std::unique_ptr<FitsColumnReader> reader);

    /**
     *  Map any remaining items into regular Schema items, and return the final Schema.
     *
     *  This method must be called before any calls to readRecords().
     */
    Schema finalize();

    /**
     *  Fill a record from a FITS binary table row.
     */
    void readRecord(BaseRecord &record, afw::fits::Fits &fits, std::size_t row);

private:
    class Impl;
    std::shared_ptr<Impl> _impl;
};
}  // namespace io
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_IO_FitsSchemaInputMapper_h_INCLUDED
