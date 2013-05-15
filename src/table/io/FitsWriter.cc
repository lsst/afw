// -*- lsst-c++ -*-

#include "boost/cstdint.hpp"

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/io/OutputArchive.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

typedef FitsWriter::Fits Fits;

//----- Code to write FITS headers --------------------------------------------------------------------------

// The driver code is at the bottom of this section; it's easier to understand if you start there
// and work your way up.

// A Schema::forEach functor that writes FITS header keys for a field when it is called.
struct ProcessSchema {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        std::string name = item.field.getName();
        std::replace(name.begin(), name.end(), '.', '_');
        int n = fits->addColumn<typename Field<T>::Element>(
            name, item.field.getElementCount(),
            item.field.getDoc()
        );
        if (!item.field.getDoc().empty()) {
            // We use a separate key TDOCn for documentation (in addition to the TTYPEn comments)
            // so we can have long strings via the CONTINUE convention.
            // When reading, if there is no TDOCn, we'll just use the TTYPEn comment.
            fits->writeColumnKey("TDOC", n, item.field.getDoc());
        }
        specialize(item, n); // delegate to other member functions that are specialized on field tag types
    }

    void operator()(SchemaItem<PTR(io::Persistable)> const & item) const {
        std::string name = item.field.getName();
        std::replace(name.begin(), name.end(), '.', '_');
        int n = fits->addColumn<int>(
            name, item.field.getElementCount(),
            item.field.getDoc()
        );
        if (!item.field.getDoc().empty()) {
            // We use a separate key TDOCn for documentation (in addition to the TTYPEn comments)
            // so we can have long strings via the CONTINUE convention.
            // When reading, if there is no TDOCn, we'll just use the TTYPEn comment.
            fits->writeColumnKey("TDOC", n, item.field.getDoc());
        }
        fits->writeColumnKey("TCCLS", n, "Persistable", "Field template used by lsst.afw.table");
    }

    void operator()(SchemaItem<std::string> const & item) const {
        std::string name = item.field.getName();
        std::replace(name.begin(), name.end(), '.', '_');
        int n = fits->addColumn<std::string>(
            name, item.field.getElementCount(),
            item.field.getDoc()
        );
        if (!item.field.getDoc().empty()) {
            fits->writeColumnKey("TDOC", n, item.field.getDoc());
        }
        specialize(item, n);
    }

    void operator()(SchemaItem<Flag> const & item) const {
        std::string name = item.field.getName();
        std::replace(name.begin(), name.end(), '.', '_');
        fits->writeColumnKey("TFLAG", nFlags, name);
        if (!item.field.getDoc().empty()) {
            // We use a separate key TFDOCn for documentation instead of the comment on TFLAGn so
            // we can have long strings via the CONTINUE convention.
            // When reading, if there is no TFDOCn, we'll use the TTYPEn comment.
            fits->writeColumnKey("TFDOC", nFlags, item.field.getDoc());
        }
        ++nFlags;
    }

    // Create and apply the functor to a schema.
    static void apply(Fits & fits, Schema const & schema) {
        ProcessSchema f = { &fits, 0 };
        schema.forEach(boost::ref(f));
    }

    template <typename T>
    void specialize(SchemaItem<T> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits());
        fits->writeColumnKey("TCCLS", n, "Scalar", "Field template used by lsst.afw.table");
    }

    void specialize(SchemaItem<Angle> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits());
        fits->writeColumnKey("TCCLS", n, "Angle", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Array<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits());
        fits->writeColumnKey("TCCLS", n, "Array", "Field template used by lsst.afw.table");
    }

    void specialize(SchemaItem<Coord> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits());
        fits->writeColumnKey("TCCLS", n, "Coord", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Point<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits(), "{x, y}");
        fits->writeColumnKey("TCCLS", n, "Point", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Moments<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits(), "{xx, yy, xy}");
        fits->writeColumnKey("TCCLS", n, "Moments", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits(),
                                 "{(0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...}");
        fits->writeColumnKey("TCCLS", n, "Covariance", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance< Point<T> > > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits(),
                                 "{(x,x), (x,y), (y,y)}");
        fits->writeColumnKey("TCCLS", n, "Covariance(Point)", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance< Moments<T> > > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits(),
                                 "{(xx,xx), (xx,yy), (yy,yy), (xx,xy), (yy,xy), (xy,xy)}");
        fits->writeColumnKey("TCCLS", n, "Covariance(Moments)", "Field template used by lsst.afw.table");
    }

    void specialize(SchemaItem< std::string > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits());
        fits->writeColumnKey("TCCLS", n, "String", "Field template used by lsst.afw.table");
    }

    Fits * fits;
    mutable int nFlags;
};

} // anonymous

// the driver for all the above machinery
void FitsWriter::_writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows) {
    Schema schema = table->getSchema();
    if (!_archive && schema.hasPersistableFields()) {
        _doWriteArchive = true;
        _fits->writeKey(
            "AR_HDU", _fits->countHdus() + 1,
            "First HDU for the binary table archive containing non-POD fields"
        );
        _archive.reset(new io::OutputArchive());
    }
    _fits->createTable();
    LSST_FITS_CHECK_STATUS(*_fits, "creating table");
    int nFlags = schema.getFlagFieldCount();
    if (nFlags > 0) {
        int n = _fits->addColumn<bool>("flags", nFlags, "bits for all Flag fields; see also TFLAGn");
        _fits->writeKey("FLAGCOL", n + 1, "Column number for the bitflags.");
    }
    ProcessSchema::apply(*_fits, schema);
    if (table->getMetadata()) {
        _fits->writeMetadata(*table->getMetadata());
    }
    _row = -1;
    _fits->addRows(nRows);
    _processor = boost::make_shared<ProcessRecords>(_fits, schema, nFlags, _row, _archive.get());
}

//----- Code for writing FITS records -----------------------------------------------------------------------

// The driver code is at the bottom of this section; it's easier to understand if you start there
// and work your way up.

// A Schema::forEach functor that writes table data for a single record when it is called.
// We instantiate one of these, then reuse it on all the records after updating the data
// members that tell it which record and row number it's on.
struct FitsWriter::ProcessRecords {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        fits->writeTableArray(row, col, item.key.getElementCount(), record->getElement(item.key));
        ++col;
    }

    void operator()(SchemaItem<std::string> const & item) const {
        fits->writeTableScalar(row, col, record->get(item.key));
        ++col;
    }

    void operator()(SchemaItem<PTR(io::Persistable)> const & item) const {
        int id = archive->put(record->get(item.key));
        fits->writeTableScalar(row, col, id);
        ++col;
    }

    void operator()(SchemaItem<Flag> const & item) const {
        flags[bit] = record->get(item.key);
        ++bit;
    }

    ProcessRecords(
        Fits * fits_,
        Schema const & schema_,
        int nFlags_,
        std::size_t const & row_,
        io::OutputArchive * archive_
    ) :
        row(row_), col(0), bit(0), nFlags(nFlags_), fits(fits_), schema(schema_), archive(archive_)
    {
        if (nFlags) flags.reset(new bool[nFlags]);
    }

    void apply(BaseRecord const * r) {
        record = r;
        col = 0;
        bit = 0;
        if (nFlags) ++col;
        schema.forEach(boost::ref(*this));
        if (nFlags) fits->writeTableArray(row, 0, nFlags, flags.get());
    }

    std::size_t const & row;
    mutable int col;
    mutable int bit;
    int nFlags;
    Fits * fits;
    boost::scoped_array<bool> flags;
    BaseRecord const * record;
    Schema schema;
    io::OutputArchive * archive;
};

void FitsWriter::_writeRecord(BaseRecord const & record) {
    ++_row;
    _processor->apply(&record);
}

FitsWriter::FitsWriter(Fits * fits, PTR(io::OutputArchive) archive) :
    _fits(fits), _row(0), _doWriteArchive(false), _archive(archive)
{}

void FitsWriter::_finish() {
    if (_doWriteArchive) {
        _archive->writeFits(*_fits);
    }
}

}}}} // namespace lsst::afw::table::io
