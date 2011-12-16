// -*- lsst-c++ -*-

#include <cstdio>

#include "fitsio.h"
extern "C" {
#include "fitsio2.h"
}

#include "boost/cstdint.hpp"

#include "lsst/afw/table/fits.h"

namespace lsst { namespace afw { namespace table { namespace fits {

//----- createFitsHeader implementation ---------------------------------------------------------------------

namespace {

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

    template <typename T>
    void specialize(SchemaItem< Array<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Array", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Point<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(), "{x, y}");
        fits->writeColumnKey("TCCLS", n, "Point", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Shape<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(), "{xx, yy, xy}");
        fits->writeColumnKey("TCCLS", n, "Shape", "Field template used by lsst.afw.table");
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
    void specialize(SchemaItem< Covariance< Shape<T> > > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(),
                                 "{(xx,xx), (xx,yy), (yy,yy), (xx,xy), (yy,xy), (xy,xy)}");
        fits->writeColumnKey("TCCLS", n, "Covariance(Shape)", "Field template used by lsst.afw.table");
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

} // anonymous

void createFitsHeader(Fits & fits, Schema const & schema, bool sanitizeNames) {
    fits.createTable();
    fits.checkStatus();
    fits.addColumn<RecordId>("id", 1, "unique ID for the record");
    if (schema.hasTree())
        fits.addColumn<RecordId>("parent", 1, "ID for the record's parent");
    int nFlags = CountFlags::apply(schema);
    if (nFlags > 0)
        fits.addColumn<bool>("flags", nFlags, "bits for all Flag fields; see also TFLAGn");
    fits.checkStatus();
    ProcessSchema::apply(fits, schema, sanitizeNames);
}

//----- writeFitsRecords implementation ---------------------------------------------------------------------

namespace {

struct ProcessData {
    
    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        this->operator()(item.key, item.key);
    }
    
    template <typename T>
    void operator()(Key<T> const & input, Key<T> const & output) const {
        fits->writeTableArray(row, col, input.getElementCount(), iter->getElementConstPtr(input));
        ++col;
    }

    void operator()(Key<Flag> const & input, Key<Flag> const & output) const {
        flags[bit] = iter->get(input);
        ++bit;
    }

    template <typename SchemaIterable>
    static void apply(
        Fits & fits, TableBase const & table, Schema const & schema, SchemaIterable const & iterable
    ) {
        bool hasTree = schema.hasTree();
        int nFlags = CountFlags::apply(schema);
        boost::scoped_array<bool> flags;
        if (nFlags)
            flags.reset(new bool[nFlags]);
        IteratorBase const end = table.end();
        ProcessData f = { 0, 0, 0, &fits, flags.get(), table.begin() };
        while (f.iter != end) {
            f.col = 0;
            f.bit = 0;
            fits.writeTableScalar(f.row, f.col++, f.iter->getId());
            if (hasTree) fits.writeTableScalar(f.row, f.col++, f.iter->getId());
            if (nFlags) ++f.col;
            iterable.forEach(boost::ref(f));
            if (nFlags) fits.writeTableArray(f.row, 1 + hasTree, nFlags, f.flags);
            ++f.row;
            ++f.iter;
        }
    }

    int row;
    mutable int col;
    mutable int bit;
    Fits * fits;
    bool * flags;
    IteratorBase iter;
};

} // anonymous

void writeFitsRecords(Fits & fits, TableBase const & table) {
    ProcessData::apply(fits, table, table.getSchema(), table.getSchema());
    fits.checkStatus();
}

void writeFitsRecords(Fits & fits, TableBase const & table, SchemaMapper const & mapper) {
    SchemaMapper mapperCopy(mapper);
    mapperCopy.sort(SchemaMapper::OUTPUT);
    ProcessData::apply(fits, table, mapperCopy.getOutputSchema(), mapperCopy);
    fits.checkStatus();
}

}}}} // namespace lsst::afw::table::fits
