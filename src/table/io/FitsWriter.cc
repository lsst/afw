// -*- lsst-c++ -*-

#include "boost/cstdint.hpp"

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/TableBase.h"
#include "lsst/afw/table/RecordBase.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

typedef FitsWriter::Fits Fits;

//----- Code to write FITS headers --------------------------------------------------------------------------

struct CountFlags {

    template <typename T>
    void operator()(SchemaItem<T> const &) const {}

    void operator()(SchemaItem<Flag> const &) const { ++n; }

    static int apply(Schema const & schema) {
        CountFlags counter = { 0 };
        schema.forEach(boost::ref(counter));
        return counter.n;
    }

    mutable int n;
};

struct ProcessSchema {

    template <typename T>
    void specialize(SchemaItem<T> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Scalar", "Field template used by lsst.afw.table");
    }

    void specialize(SchemaItem<Angle> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Angle", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Array<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Array", "Field template used by lsst.afw.table");
    }

    void specialize(SchemaItem<Coord> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Coord", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Point<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(), "{x, y}");
        fits->writeColumnKey("TCCLS", n, "Point", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Moments<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(), "{xx, yy, xy}");
        fits->writeColumnKey("TCCLS", n, "Moments", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(),
                                 "{(0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...}");
        fits->writeColumnKey("TCCLS", n, "Covariance", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance< Point<T> > > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(),
                                 "{(x,x), (x,y), (y,y)}");
        fits->writeColumnKey("TCCLS", n, "Covariance(Point)", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance< Moments<T> > > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(),
                                 "{(xx,xx), (xx,yy), (yy,yy), (xx,xy), (yy,xy), (xy,xy)}");
        fits->writeColumnKey("TCCLS", n, "Covariance(Moments)", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        std::string name = item.field.getName();
        if (sanitizeNames)
            std::replace(name.begin(), name.end(), '.', '_');
        int n = fits->addColumn<typename Field<T>::Element>(
            name.c_str(),
            item.field.getElementCount(),
            item.field.getDoc().c_str()
        );
        specialize(item, n);
    }

    void operator()(SchemaItem<Flag> const & item) const {
        std::string name = item.field.getName();
        if (sanitizeNames)
            std::replace(name.begin(), name.end(), '.', '_');
        fits->writeColumnKey("TFLAG", nFlags, name.c_str(), item.field.getDoc().c_str());
        ++nFlags;
    }

    static void apply(Fits & fits, Schema const & schema, bool sanitizeNames) {
        ProcessSchema f = { &fits, sanitizeNames, 0 };
        schema.forEach(boost::ref(f));
    }

    Fits * fits;
    bool sanitizeNames;
    mutable int nFlags;
};

//----- code for writing FITS records -----------------------------------------------------------------------

struct ProcessWriteData {
    
    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        fits->writeTableArray(row, col, item.key.getElementCount(), record->getElement(item.key));
        ++col;
    }
    
    void operator()(SchemaItem<Flag> const & item) const {
        flags[bit] = record->get(item.key);
        ++bit;
    }

    static void apply(
        Fits & fits, RecordSource & source, Schema const & schema
    ) {
        int nFlags = CountFlags::apply(schema);
        boost::scoped_array<bool> flags;
        if (nFlags)
            flags.reset(new bool[nFlags]);
        ProcessWriteData f = { 0, 0, 0, &fits, flags.get(), source() };
        while (f.record) {
            f.col = 0;
            f.bit = 0;
            if (nFlags) ++f.col;
            schema.forEach(boost::ref(f));
            if (nFlags) fits.writeTableArray(f.row, 0, nFlags, f.flags);
            ++f.row;
            f.record = source();
        }
    }

    std::size_t row;
    mutable int col;
    mutable int bit;
    Fits * fits;
    bool * flags;
    CONST_PTR(RecordBase) record;
};

} // anonymous

void FitsWriter::_write(CONST_PTR(TableBase) const & table, RecordSource & source) {
    Schema const & schema = table->getSchema();
    _fits->createTable();
    _fits->checkStatus();
    int nFlags = CountFlags::apply(schema);
    if (nFlags > 0) {
        int n = _fits->addColumn<bool>("flags", nFlags, "bits for all Flag fields; see also TFLAGn");
        _fits->writeKey("FLAGCOL", n, "Column number for the bitflags.");
    }
    ProcessSchema::apply(*_fits, schema, true);
    ProcessWriteData::apply(*_fits, source, schema);
    _fits->checkStatus();
    _fits->checkStatus();
}

}}}} // namespace lsst::afw::table::io
