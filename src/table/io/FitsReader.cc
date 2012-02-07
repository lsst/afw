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
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

typedef FitsReader::Fits Fits;

/*
 *  This file contains most of the code for reading FITS binary tables.  There's also a little
 *  in Source.cc, where we read the stuff that's specific to SourceTable/SourceRecord (footprints
 *  and aliases).
 */

// ------------ FITS header to Schema implementation -------------------------------------------------------

/*
 *  We read FITS headers in two stages - first we read all the information we care about into
 *  a temporary structure (FitsSchema) we can access more easily than a raw FITS header,
 *  and then we iterate through that to fill the actual Schema object.
 *
 *  The driver code is at the bottom of this section; it's easier to understand if you start there
 *  and work your way up.
 *
 */

// A structure that describes a field as a bunch of strings read from the FITS header.
struct FitsSchemaItem {
    int col;             // column number (0-indexed)
    int bit;             // bit number for flag fields; -1 for others.
    std::string name;    // name of the field (from TTYPE keys)
    std::string units;   // field units (from TUNIT keys)
    std::string doc;     // field docs (from comments on TTYPE keys)
    std::string format;  // FITS column format code (from TFORM keys)
    std::string cls;     // which field class to use (from our own TCCLS keys)

    // Add the field defined by the strings to a schema.
    void addField(Fits & fits, Schema & schema) const {
        static boost::regex const regex("(\\d*)(\\u)(\\d)*", boost::regex::perl);
        // start by parsing the format; this tells the element type of the field and the number of elements
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
        // switch code over FITS codes that correspond to different element types
        switch (code) {
        case 'J': // 32-bit integers - can only be scalars or Point fields.
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
        case 'K': // 64-bit integers - can only be scalars.
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
        case 'E': // floats and doubles can be any number of things; delegate to a separate function
            addFloatField<float>(fits, schema, size);
            break;
        case 'D':
            addFloatField<double>(fits, schema, size);
            break;
        default:
            // We throw if we encounter a column type we can't handle.
            // This raises probem when we want to save footprints as variable length arrays
            // later, so we add the nCols argument to Reader::_readSchema to allow SourceFitsReader
            // to call FitsReader::_readSchema in a way that prevents it from ever getting here.
            throw LSST_EXCEPT(
                afw::fits::FitsError,
                afw::fits::makeErrorMessage(
                    fits.fptr, fits.status,
                    boost::format("Unsupported FITS column type: '%s'.") % format
                )
            );
        }
    }

    // Add a field with a float or double element type to the schema.
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


    FitsSchemaItem(int col_, int bit_) : col(col_), bit(bit_) {}
};

// A quirk of Boost.MultiIndex (which we use for our container of FitsSchemaItems)
// that you have to use a special functor (like this one) to set data members
// in a container with set indices (because setting those values might require 
// the element to be moved to a different place in the set).  Check out
// the Boost.MultiIndex docs for more information.
template <std::string FitsSchemaItem::*Member>
struct SetFitsSchemaString {
    void operator()(FitsSchemaItem & item) {
        item.*Member = _v;
    }
    explicit SetFitsSchemaString(std::string const & v) : _v(v) {}
private:
    std::string const & _v;
};

// A container class (based on Boost.MultiIndex) that provides two sort orders,
// on column number and on flag bit.  This allows us to insert fields into the
// schema in the correct order, regardless of which order they appear in the
// FITS header.
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

    // Typedefs for the special functors used to set data members.
    typedef SetFitsSchemaString<&FitsSchemaItem::name> SetName;
    typedef SetFitsSchemaString<&FitsSchemaItem::units> SetUnits;
    typedef SetFitsSchemaString<&FitsSchemaItem::doc> SetDoc;
    typedef SetFitsSchemaString<&FitsSchemaItem::format> SetFormat;
    typedef SetFitsSchemaString<&FitsSchemaItem::cls> SetCls;

    // Typedefs for the different indices.
    typedef Container::nth_index<0>::type ColSet;
    typedef Container::nth_index<1>::type BitSet;
    typedef Container::nth_index<2>::type List;

    // Getters for the different indices.
    ColSet & asColSet() { return container.get<0>(); }
    BitSet & asBitSet() { return container.get<1>(); }
    List & asList() { return container.get<2>(); }

    Container container;
};

// This is a polymorphic functor that's passed to Fits::forEachKey.  That calls
// the functor with the keyword name, the keyword value, and the comment for each
// key in the FITS header.  We use it to fill the FitsSchema multi-index container.
struct ProcessHeader : public afw::fits::HeaderIterationFunctor {

    virtual void operator()(std::string const & key, std::string const & value, std::string const & comment) {
        if (key.compare(0, 5, "TTYPE") == 0) {
            int col = boost::lexical_cast<int>(key.substr(5)) - 1;
            FitsSchema::ColSet::iterator i = schema.asColSet().lower_bound(col);
            if (i == schema.asColSet().end() || i->col != col) {
                i = schema.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            std::string v = value;
            std::replace(v.begin(), v.end(), '_', '.');
            schema.asColSet().modify(i, FitsSchema::SetName(v));
            schema.asColSet().modify(i, FitsSchema::SetDoc(comment));
        } else if (key.compare(0, 5, "TFLAG") == 0) {
            int bit = boost::lexical_cast<int>(key.substr(5)) - 1;
            FitsSchema::BitSet::iterator i = schema.asBitSet().lower_bound(bit);
            if (i == schema.asBitSet().end() || i->bit != bit) {
                i = schema.asBitSet().insert(i, FitsSchemaItem(-1, bit));
            }
            std::string v = value;
            std::replace(v.begin(), v.end(), '_', '.');
            schema.asBitSet().modify(i, FitsSchema::SetName(v));
            schema.asBitSet().modify(i, FitsSchema::SetDoc(comment));
        } else if (key.compare(0, 5, "TUNIT") == 0) {
            int col = boost::lexical_cast<int>(key.substr(5)) - 1;
            FitsSchema::ColSet::iterator i = schema.asColSet().lower_bound(col);
            if (i == schema.asColSet().end() || i->col != col) {
                i = schema.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            schema.asColSet().modify(i, FitsSchema::SetUnits(value));
        } else if (key.compare(0, 5, "TCCLS") == 0) {
            int col = boost::lexical_cast<int>(key.substr(5)) - 1;
            FitsSchema::ColSet::iterator i = schema.asColSet().lower_bound(col);
            if (i == schema.asColSet().end() || i->col != col) {
                i = schema.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            schema.asColSet().modify(i, FitsSchema::SetCls(value));
        } else if (key.compare(0, 5, "TFORM") == 0) {
            int col = boost::lexical_cast<int>(key.substr(5)) - 1;
            FitsSchema::ColSet::iterator i = schema.asColSet().lower_bound(col);
            if (i == schema.asColSet().end() || i->col != col) {
                i = schema.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            schema.asColSet().modify(i, FitsSchema::SetFormat(value));
        } else if (key.compare(0, 8, "FLAGCOL") == 0) {
            flagCol = boost::lexical_cast<int>(value) - 1;
        }
    }

    explicit ProcessHeader() : schema(), flagCol(-1) {}

    FitsSchema schema;
    int flagCol;
};

} // anonymous

// finally, here's the driver for all of the machinery above
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

PTR(BaseTable) FitsReader::_readTable(Schema const & schema) {
    return BaseTable::make(schema);
}

// ------------ FITS table to records implementation --------------------------------------------------------

/*
 *  Compared to reading the header, reading the records is pretty easy.  We just
 *  create a Schema::forEach functor (ProcessRecords) and have the schema use
 *  it to iterate over all the fields for each record.  We actually create that
 *  object above, in _readSchema, and then call Schema::forEach in _readRecord.
 *
 *  The driver code is at the bottom of this section; it's easier to understand if you start there
 *  and work your way up.
 */

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
            --flagCol; // we want 0-indexed column numbers, not FITS' 1-indexed numbers
        } else {
            fits->status = 0;
            flagCol = -1;
        }
        nFlags = fits->getTableArraySize(flagCol);
        if (nFlags) flags.reset(new bool[nFlags]);
    }

    std::size_t const & row;  // this is a reference back to the _row data member in FitsReader
    mutable int col;          // the current column (0-indexed)
    mutable int bit;          // the current flag bit (in the FITS table, not the Schema)
    int nFlags;               // the total number of flags
    int flagCol;              // the column number (0-indexed) that holds all the flag bits
    Fits * fits;              // the FITS file pointer
    boost::scoped_array<bool> flags;  // space to hold a bool array of the flags to pass to cfitsio
    BaseRecord * record;      // record to write values to
};

PTR(BaseRecord) FitsReader::_readRecord(PTR(BaseTable) const & table) {
    PTR(BaseRecord) record;
    if (++_row == _nRows) return record;
    record = table->makeRecord();
    _processor->record = record.get();
    _processor->apply(table->getSchema());
    return record;
}

// ------------ FitsReader Registry implementation ----------------------------------------------------------

namespace {

typedef std::map<std::string,FitsReader::Factory*> Registry;

Registry & getRegistry() {
    static Registry it;
    return it;
}

// here's an example of how you register a FitsReader
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

}}}} // namespace lsst::afw::table::io
