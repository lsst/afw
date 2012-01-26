// -*- lsst-c++ -*-

#include <cstdio>

#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/cstdint.hpp"
#include "boost/multi_index_container.hpp"
#include "boost/multi_index/sequenced_index.hpp"
#include "boost/multi_index/ordered_index.hpp"
#include "boost/multi_index/member.hpp"
#include "boost/math/special_functions/round.hpp"

#include "lsst/afw/table/io/FitsReader.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/TableBase.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

typedef FitsReader::Fits Fits;

/*
 *  We read FITS headers in two stages - first we read all the information we care about into
 *  a temporary structure (FitsSchema) we can access more easily than a raw FITS header,
 *  and then we iterate through that to fill the actual Schema object.
 */

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
            if (cls == "Angle") {
                schema.addField< Angle >(name, doc, units);
            } else if (cls == "Array") {
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
            if (cls == "Coord") {
                schema.addField< Coord >(name, doc, units);
                return;
            }
        } else if (size == 3) {
            if (cls == "Moments") {
                schema.addField< Moments<U> >(name, doc, units);
                return;
            }
            if (cls == "Covariance(Point)") {
                schema.addField< Covariance< Point<U> > >(name, doc, units);
                return;
            }
        } else if (size == 6) {
            if (cls == "Covariance(Moments)") {
                schema.addField< Covariance< Moments<U> > >(name, doc, units);
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
        } else if (std::strncmp(key, "FLAGCOL", 8) == 0) {
            flagCol = boost::lexical_cast<int>(value) - 1;
        }
    }

    explicit ProcessHeader() : schema(), flagCol(-1) {}

    FitsSchema schema;
    int flagCol;
};

typedef std::map<std::string,FitsReader::Factory*> Registry;

Registry & getRegistry() {
    static Registry it;
    return it;
}

static FitsReader::FactoryT<FitsReader> baseReaderFactory("BASE");

} // anonymous

FitsReader::Factory::Factory(std::string const & name) {
    getRegistry()[name] = this;
}

PTR(FitsReader) FitsReader::make(Fits * fits) {
    std::string name;
    fits->readKey("AFW_TYPE", name);
    if (fits->status != 0) {
        name = "BASE";
        fits->status = 0;
    }
    Registry::iterator i = getRegistry().find(name);
    if (i == getRegistry().end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            (boost::format("FitsReader with name '%s' does not exist; check AFW_TYPE keyword.") % name).str()
        );
    }
    return (*i->second)(fits);
}

struct FitsReader::ProcessRecords {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        if (col == flagCol) ++col;
        fits->readTableArray(row, col, item.key.getElementCount(), record->getElement(item.key));
        ++col;
    }

    void operator()(SchemaItem<Flag> const & item) const {
        record->set(item.key, flags[bit]);
        ++bit;
    }

    void apply(Schema const & schema) {
        col = 0;
        bit = 0;
        if (flagCol >= 0) {
            fits->readTableArray<bool>(row, flagCol, nFlags, flags.get());
        }
        schema.forEach(boost::ref(*this));
    }

    ProcessRecords(Fits * fits_, std::size_t const & row_) :
        row(row_), col(0), bit(0), flagCol(-1), fits(fits_)
    {
        fits->readKey("FLAGCOL", flagCol);
        if (fits->status == 0) {
            --flagCol;
        } else {
            fits->status = 0;
            flagCol = -1;
        }
        nFlags = fits->getTableArraySize(flagCol);
        if (nFlags) flags.reset(new bool[nFlags]);
    }

    std::size_t const & row;
    mutable int col;
    mutable int bit;
    int nFlags;
    int flagCol;
    Fits * fits;
    boost::scoped_array<bool> flags;
    RecordBase * record;
};

Schema FitsReader::_readSchema(int nCols) {
    ProcessHeader f;
    _fits->forEachKey(f);
    Schema schema;
    for (
        FitsSchema::List::const_iterator i = f.schema.asList().begin();
        i != f.schema.asList().end();
        ++i
    ) {
        if (nCols >= 0 && i->col >= nCols) continue;
        if (i->bit >= 0) {
            schema.addField<Flag>(i->name, i->doc);
        } else if (i->col != f.flagCol) {
            i->addField(*_fits, schema);
        }
    }
    _row = -1;
    _nRows = _fits->countRows();
    _processor = boost::make_shared<ProcessRecords>(_fits, _row);
    return schema;
}

PTR(TableBase) FitsReader::_readTable(Schema const & schema) {
    return TableBase::make(schema);
}

PTR(RecordBase) FitsReader::_readRecord(PTR(TableBase) const & table) {
    PTR(RecordBase) record;
    if (++_row == _nRows) return record;
    record = table->makeRecord();
    _processor->record = record.get();
    _processor->apply(table->getSchema());
    return record;
}

}}}} // namespace lsst::afw::table::io
