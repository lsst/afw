// -*- lsst-c++ -*-

#include <cstdio>

#include "fitsio.h"
extern "C" {
#include "fitsio2.h"
}

#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/cstdint.hpp"
#include "boost/multi_index_container.hpp"
#include "boost/multi_index/sequenced_index.hpp"
#include "boost/multi_index/ordered_index.hpp"
#include "boost/multi_index/member.hpp"
#include "boost/math/special_functions/round.hpp"

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

struct ProcessWriteData {
    
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
        ProcessWriteData f = { 0, 0, 0, &fits, flags.get(), table.begin() };
        while (f.iter != end) {
            f.col = 0;
            f.bit = 0;
            fits.writeTableScalar(f.row, f.col++, f.iter->getId());
            if (hasTree) fits.writeTableScalar(f.row, f.col++, f.iter->getParentId());
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
    ProcessWriteData::apply(fits, table, table.getSchema(), table.getSchema());
    fits.checkStatus();
}

void writeFitsRecords(Fits & fits, TableBase const & table, SchemaMapper const & mapper) {
    SchemaMapper mapperCopy(mapper);
    mapperCopy.sort(SchemaMapper::OUTPUT);
    ProcessWriteData::apply(fits, table, mapperCopy.getOutputSchema(), mapperCopy);
    fits.checkStatus();
}

//----- readFitsHeader implementation -----------------------------------------------------------------------

/*
 *  We read FITS headers in two stages - first we read all the information we care about into
 *  a temporary structure (FitsSchema) we can access more easily than a raw FITS header,
 *  and then we iterate through that to fill the actual Schema object.
 */

namespace {

std::string strip(std::string const & s) {
    std::size_t i1 = s.find_first_not_of(" '");
    std::size_t i2 = s.find_last_not_of(" '");
    return s.substr(i1, (i1 == std::string::npos) ? 0 : 1 + i2 - i1);
}

struct FitsSchemaItem {
    int col;
    int bit;
    std::string name;
    std::string units;
    std::string doc;
    std::string format;
    std::string cls;

    template <typename U>
    void addFloatField(Fits & fits, Schema & schema, int size) const {
        if (size == 1) {
            if (cls == "Array") {
                schema.addField< Array<U> >(name, doc, units, 1);
            } else if (cls == "Covariance") {
                schema.addField< Covariance<U> >(name, doc, units, 1);
            } else {
                schema.addField<U>(name, doc, units);
            }
            return;
        } else if (size == 2) {
            if (cls == "Point") {
                schema.addField< Point<U> >(name, doc, units);
                return;
            }
        } else if (size == 3) {
            if (cls == "Shape") {
                schema.addField< Shape<U> >(name, doc, units);
                return;
            }
            if (cls == "Covariance(Point)") {
                schema.addField< Covariance< Point<U> > >(name, doc, units);
                return;
            }
        } else if (size == 6) {
            if (cls == "Covariance(Shape)") {
                schema.addField< Covariance< Shape<U> > >(name, doc, units);
                return;
            }
        }
        if (cls == "Covariance") {
            double v = 0.5 * (std::sqrt(1 + 8 * size) - 1);
            int n = boost::math::iround(v);
            if (n * (n + 1) != size * 2) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        fits.fptr,
                        fits.status,
                        boost::format("Covariance field has invalid size.")
                    )
                );
            }
            schema.addField< Covariance<U> >(name, doc, units, n);
        } else {
            schema.addField< Array<U> >(name, doc, units, size);
        }
    }

    void addField(Fits & fits, Schema & schema) const {
        static boost::regex const regex("(\\d*)(\\u)(\\d)*", boost::regex::perl);
        boost::smatch m;
        if (!boost::regex_match(format, m, regex)) {
            throw LSST_EXCEPT(
                afw::fits::FitsError,
                afw::fits::makeErrorMessage(
                    fits.fptr, fits.status,
                    boost::format("Could not parse TFORM value for field '%s': '%s'.") % name % format
                )
            );
        }
        int size = 1;
        if (m[1].matched)
            size = boost::lexical_cast<int>(m[1].str());        
        char code = m[2].str()[0];
        switch (code) {
        case 'J':
            if (size == 1) {
                schema.addField<boost::int32_t>(name, doc, units);
            } else if (size == 2) {
                schema.addField< Point<boost::int32_t> >(name, doc, units);
            } else {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        fits.fptr, fits.status,
                        boost::format("Unsupported FITS column type: '%s'.") % format
                    )
                );
            }
            break;
        case 'K':
            if (size != 1) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        fits.fptr, fits.status,
                        boost::format("Unsupported FITS column type: '%s'.") % format
                    )
                );
            }
            schema.addField<boost::int64_t>(name, units, doc);
            break;
        case 'E':
            addFloatField<float>(fits, schema, size);
            break;
        case 'D':
            addFloatField<double>(fits, schema, size);
            break;
        default:
            throw LSST_EXCEPT(
                afw::fits::FitsError,
                afw::fits::makeErrorMessage(
                    fits.fptr, fits.status,
                    boost::format("Unsupported FITS column type: '%s'.") % format
                )
            );
        }
    }

    FitsSchemaItem(int col_, int bit_) : col(col_), bit(bit_) {}
};

template <std::string FitsSchemaItem::*Member>
struct SetFitsSchemaString {
    void operator()(FitsSchemaItem & item) {
        item.*Member = _v;
    }
    explicit SetFitsSchemaString(std::string const & v) : _v(v) {}
private:
    std::string const & _v;
};

struct FitsSchema {
    typedef boost::multi_index_container<
        FitsSchemaItem,
        boost::multi_index::indexed_by<
            boost::multi_index::ordered_non_unique<
                boost::multi_index::member<FitsSchemaItem,int,&FitsSchemaItem::col>
                >,
            boost::multi_index::ordered_non_unique<
                boost::multi_index::member<FitsSchemaItem,int,&FitsSchemaItem::bit>
                >,
            boost::multi_index::sequenced<>
            >
        > Container;

    typedef SetFitsSchemaString<&FitsSchemaItem::name> SetName;
    typedef SetFitsSchemaString<&FitsSchemaItem::units> SetUnits;
    typedef SetFitsSchemaString<&FitsSchemaItem::doc> SetDoc;
    typedef SetFitsSchemaString<&FitsSchemaItem::format> SetFormat;
    typedef SetFitsSchemaString<&FitsSchemaItem::cls> SetCls;

    typedef Container::nth_index<0>::type ColSet;
    typedef Container::nth_index<1>::type BitSet;
    typedef Container::nth_index<2>::type List;

    ColSet & asColSet() { return container.get<0>(); }
    BitSet & asBitSet() { return container.get<1>(); }
    List & asList() { return container.get<2>(); }

    Container container;
};

struct ProcessHeader : public afw::fits::HeaderIterationFunctor {

    virtual void operator()(char const * key, char const * value, char const * comment) {
        if (std::strncmp(key, "TTYPE", 5) == 0) {
            int col = boost::lexical_cast<int>(key + 5) - 1;
            FitsSchema::ColSet::iterator i = schema.asColSet().lower_bound(col);
            if (i == schema.asColSet().end() || i->col != col) {
                i = schema.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            schema.asColSet().modify(i, FitsSchema::SetName(strip(value)));
            schema.asColSet().modify(i, FitsSchema::SetDoc(comment));
        } else if (std::strncmp(key, "TFLAG", 5) == 0) {
            int bit = boost::lexical_cast<int>(key + 5) - 1;
            FitsSchema::BitSet::iterator i = schema.asBitSet().lower_bound(bit);
            if (i == schema.asBitSet().end() || i->bit != bit) {
                i = schema.asBitSet().insert(i, FitsSchemaItem(-1, bit));
            }
            schema.asBitSet().modify(i, FitsSchema::SetName(strip(value)));
            schema.asBitSet().modify(i, FitsSchema::SetDoc(comment));
        } else if (std::strncmp(key, "TUNIT", 5) == 0) {
            int col = boost::lexical_cast<int>(key + 5) - 1;
            FitsSchema::ColSet::iterator i = schema.asColSet().lower_bound(col);
            if (i == schema.asColSet().end() || i->col != col) {
                i = schema.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            schema.asColSet().modify(i, FitsSchema::SetUnits(strip(value)));
        } else if (std::strncmp(key, "TCCLS", 5) == 0) {
            int col = boost::lexical_cast<int>(key + 5) - 1;
            FitsSchema::ColSet::iterator i = schema.asColSet().lower_bound(col);
            if (i == schema.asColSet().end() || i->col != col) {
                i = schema.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            schema.asColSet().modify(i, FitsSchema::SetCls(strip(value)));
        } else if (std::strncmp(key, "TFORM", 5) == 0) {
            int col = boost::lexical_cast<int>(key + 5) - 1;
            FitsSchema::ColSet::iterator i = schema.asColSet().lower_bound(col);
            if (i == schema.asColSet().end() || i->col != col) {
                i = schema.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            schema.asColSet().modify(i, FitsSchema::SetFormat(strip(value)));
        } else if (std::strncmp(key, "ID_COL", 6) == 0) {
            idCol = boost::lexical_cast<int>(value) - 1;
        } else if (std::strncmp(key, "TREE_COL", 8) == 0) {
            treeCol = boost::lexical_cast<int>(value) - 1;
        } else if (std::strncmp(key, "FLAG_COL", 8) == 0) {
            flagCol = boost::lexical_cast<int>(value) - 1;
        }
    }

    explicit ProcessHeader() : schema(), idCol(-1), treeCol(-1), flagCol(-1) {}

    FitsSchema schema;
    int idCol;
    int treeCol;
    int flagCol;
};

} // anonymous

Schema readFitsHeader(Fits & fits, bool unsanitizeNames, int nCols) {
    ProcessHeader f;
    fits.forEachKey(f);
    Schema schema(f.treeCol >= 0);
    for (
        FitsSchema::List::const_iterator i = f.schema.asList().begin();
        i != f.schema.asList().end();
        ++i
    ) {
        if (nCols >= 0 && i->col >= nCols) continue;
        if (i->bit >= 0) {
            schema.addField<Flag>(i->name, i->doc);
        } else if (i->col != f.idCol && i->col != f.treeCol && i->col != f.flagCol) {
            i->addField(fits, schema);
        }
    }
    return schema;
}

//----- readFitsRecords implementation ----------------------------------------------------------------------

namespace {

struct ProcessReadData {
    
    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        this->operator()(item.key, item.key);
    }

    template <typename T>
    void operator()(Key<T> const & input, Key<T> const & output) const {
        while (col == idCol || col == treeCol || col == flagCol) ++col;
        fits->readTableArray(row, col, input.getElementCount(), record->getElementPtr(input));
        ++col;
    }

    void operator()(Key<Flag> const & input, Key<Flag> const & output) const {
        record->set(input, flags[bit]);
        ++bit;
    }

    template <typename SchemaIterable>
    void doRecord(SchemaIterable const & iterable, int nFlags) {
        if (treeCol >= 0) {
            RecordId parentId;
            fits->readTableScalar(row, treeCol, parentId);
            record->setParentId(parentId);
        }
        if (flagCol >= 0) {
            fits->readTableArray<bool>(row, flagCol, nFlags, flags);
        }
        iterable.forEach(*this);
    }

    template <typename SchemaIterable>
    static void apply(
        Fits & fits, TableBase const & table,
        Schema const & schema, SchemaIterable const & iterable,
        int idCol, int treeCol, int flagCol
    ) {
        int nFlags = CountFlags::apply(schema);
        boost::scoped_array<bool> flags;
        if (nFlags)
            flags.reset(new bool[nFlags]);
        int nRows = fits.countRows();
        ProcessReadData f = { 0, 0, 0, &fits, flags.get(), 0, idCol, treeCol, flagCol };
        while (f.row < nRows) {
            RecordId recordId = 0;
            f.col = 0;
            f.bit = 0;
            if (idCol >= 0) {
                fits.readTableScalar(f.row, idCol, recordId);
                RecordBase record = detail::Access::addRecord(table, recordId);
                f.record = &record;
                f.doRecord(iterable, nFlags);
            } else {
                RecordBase record = detail::Access::addRecord(table);
                f.record = &record;
                f.doRecord(iterable, nFlags);
            }
            ++f.row;
        }
    }

    int row;
    mutable int col;
    mutable int bit;
    Fits * fits;
    bool * flags;
    RecordBase * record;
    int idCol;
    int treeCol;
    int flagCol;
};

} // anonymous

void readFitsRecords(Fits & fits, TableBase const & table) {
    int idCol = -1, treeCol = -1, flagCol = -1;
    fits.readKey("ID_COL", idCol);
    if (fits.status == 0) {
        --idCol;
    } else {
        fits.status = 0;
        idCol = -1;
    }
    fits.readKey("FLAG_COL", flagCol);
    if (fits.status == 0) {
        --flagCol;
    } else {
        fits.status = 0;
        flagCol = -1;
    }
    fits.readKey("TREE_COL", treeCol);
    if (fits.status == 0) {
        --treeCol;
    } else {
        fits.status = 0;
        treeCol = -1;
    }    
    ProcessReadData::apply(fits, table, table.getSchema(), table.getSchema(), idCol, treeCol, flagCol);
    fits.checkStatus();
}

}}}} // namespace lsst::afw::table::fits
