// -*- lsst-c++ -*-

#include <cstdio>

#include "fitsio.h"
extern "C" {
#include "fitsio2.h"
}

#include "boost/lexical_cast.hpp"
#include "boost/cstdint.hpp"

#include "lsst/afw/table/fits.h"

namespace lsst { namespace afw { namespace table { namespace fits {

//----- writeFitsHeader implementation ---------------------------------------------------------------------

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

void writeFitsHeader(Fits & fits, Schema const & schema, bool sanitizeNames) {
    fits.createTable();
    fits.checkStatus();
    int n = fits.addColumn<RecordId>("id", 1, "unique ID for the record");
    fits.writeKey("ID_COL", n+1, "Number of the column with a unique ID.");
    if (schema.hasTree()) {
        n = fits.addColumn<RecordId>("parent", 1, "ID for the record's parent");
        fits.writeKey("TREE_COL", n + 1, "Number of the column with tree IDs.");
    }
    int nFlags = CountFlags::apply(schema);
    if (nFlags > 0) {
        n = fits.addColumn<bool>("flags", nFlags, "bits for all Flag fields; see also TFLAGn");
        fits.writeKey("FLAG_COL", n + 1, "Number of the column bitflags.");
    }
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

//----- readFitsHeader implementation -----------------------------------------------------------------------

namespace {

std::string strip(std::string const & s) {
    std::size_t i1 = s.find_first_not_of(" '");
    std::size_t i2 = s.find_last_not_of(" '");
    return s.substr(i1, (i1 == std::string::npos) ? 0 : 1 + i2 - i1);
}

struct HeaderParser : public afw::fits::HeaderIterationFunctor {

    virtual void operator()(char const * key, char const * value, char const * comment) {
        if (std::strncmp(key, "TTYPE", 5) == 0) {
            int n = boost::lexical_cast<int>(key + 5) - 1;
            if (n != col) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        fits->fptr, 0,
                        boost::format("Out of sequence TTYPE%d key (should be TTYPE%d).") % (n+1) % (col+1)
                    )
                );
            }
            if (n == idCol || n == treeCol || n == flagCol) return; // these are handled specially
            std::string name = strip(value);
            ++col;
        } else if (std::strncmp(key, "TFLAG", 5) == 0) {
            int n = boost::lexical_cast<int>(key + 5) - 1;
            std::string name = strip(value);
            ++bit;
        }
    }

    int idCol;
    int treeCol;
    int flagCol;
    mutable int col;
    mutable int bit;
    Schema * schema;
    Fits * fits;
};

} // anonymous

Schema readFitsHeader(Fits & fits, bool unsanitizeNames) {


    HeaderParser headerParser;
    fits.forEachKey(headerParser);
    Schema schema(false);    
    return schema;
}

}}}} // namespace lsst::afw::table::fits
