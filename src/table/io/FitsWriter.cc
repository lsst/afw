// -*- lsst-c++ -*-

#include <memory>

#include "boost/cstdint.hpp"

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/BaseRecord.h"

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

    void operator()(SchemaItem<std::string> const & item) const {
        std::string name = item.field.getName();
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

    void specialize(SchemaItem< std::string > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits());
        fits->writeColumnKey("TCCLS", n, "String", "Field template used by lsst.afw.table");
    }

    Fits * fits;
    mutable int nFlags;
};

void writeAliasMap(Fits & fits, AliasMap const & aliases) {
    for (AliasMap::Iterator i = aliases.begin(); i != aliases.end(); ++i) {
        fits.writeKey("ALIAS", i->first + ":" + i->second);
    }
}

} // anonymous

// the driver for all the above machinery
void FitsWriter::_writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows) {
    Schema schema = table->getSchema();
    _fits->createTable();
    LSST_FITS_CHECK_STATUS(*_fits, "creating table");
    int nFlags = schema.getFlagFieldCount();
    if (nFlags > 0) {
        int n = _fits->addColumn<bool>("flags", nFlags, "bits for all Flag fields; see also TFLAGn");
        _fits->writeKey("FLAGCOL", n + 1, "Column number for the bitflags.");
    }
    ProcessSchema::apply(*_fits, schema);
    writeAliasMap(*_fits, *schema.getAliasMap());
    // write the version number to the fits header, plus any other metadata
    PTR(daf::base::PropertyList) metadata = table->getMetadata();
    if (!metadata) {
        metadata = std::make_shared<daf::base::PropertyList>();
    }
    metadata->set<int>("AFW_TABLE_VERSION", Schema::VERSION);
    _fits->writeMetadata(*metadata);
    // In case the metadata was attached to the table, clean it up.
    metadata->remove("AFW_TABLE_VERSION");
    _row = -1;
    _fits->addRows(nRows);
    _processor = std::make_shared<ProcessRecords>(_fits, schema, nFlags, _row);
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

    template <typename T>
    void operator()(SchemaItem< Array<T> > const & item) const {
        if (item.key.isVariableLength()) {
            ndarray::Array<T const,1,1> array = record->get(item.key);
            fits->writeTableArray(row, col, array.template getSize<0>(), array.getData());
        } else {
            fits->writeTableArray(row, col, item.key.getElementCount(), record->getElement(item.key));
        }
        ++col;
    }

    void operator()(SchemaItem<std::string> const & item) const {
        fits->writeTableScalar(row, col, record->get(item.key));
        ++col;
    }
    
    void operator()(SchemaItem<Flag> const & item) const {
        flags[bit] = record->get(item.key);
        ++bit;
    }

    ProcessRecords(Fits * fits_, Schema const & schema_, int nFlags_, std::size_t const & row_) :
        row(row_), col(0), bit(0), nFlags(nFlags_), fits(fits_), schema(schema_)
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
    std::unique_ptr<bool[]> flags;
    BaseRecord const * record;
    Schema schema;
};

void FitsWriter::_writeRecord(BaseRecord const & record) {
    ++_row;
    _processor->apply(&record);
}

}}}} // namespace lsst::afw::table::io
