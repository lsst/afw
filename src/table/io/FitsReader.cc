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
    void addField(Schema & schema) const {
        static boost::regex const regex("(\\d+)?(\\u)(\\d)*", boost::regex::perl);
        // start by parsing the format; this tells the element type of the field and the number of elements
        boost::smatch m;
        if (!boost::regex_match(format, m, regex)) {
            throw LSST_EXCEPT(
                afw::fits::FitsError,
                (boost::format("Could not parse TFORM value for field '%s': '%s'.") % name % format).str()
            );
        }
        int size = 1;
        if (m[1].matched)
            size = boost::lexical_cast<int>(m[1].str());        
        char code = m[2].str()[0];
        // switch code over FITS codes that correspond to different element types
        switch (code) {
        case 'J': // 32-bit integers - can only be scalars, Point fields, or Arrays
            if (size == 1) {
                if (cls == "Array") {
                    schema.addField< Array<boost::int32_t> >(name, doc, units, size);
                } else { 
                    schema.addField<boost::int32_t>(name, doc, units);
                }
            } else if (size == 2) {
                if (cls == "Array") {
                    schema.addField< Array<boost::int32_t> >(name, doc, units, size);
                } else {
                    schema.addField< Point<boost::int32_t> >(name, doc, units);
                }
            } else {
                schema.addField< Array<boost::int32_t> >(name, doc, units, size);
            }
            break;
        case 'K': // 64-bit integers - can only be scalars.
            if (size != 1) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    (boost::format("Unsupported FITS column type: '%s'.") % format).str()
                );
            }
            schema.addField<boost::int64_t>(name, doc, units);
            break;
        case 'E': // floats and doubles can be any number of things; delegate to a separate function
            addFloatField<float>(schema, size);
            break;
        case 'D':
            addFloatField<double>(schema, size);
            break;
        case 'A': // strings
            schema.addField<std::string>(name, doc, units, size);
            break;
        default:
            // We throw if we encounter a column type we can't handle.
            // This raises probem when we want to save footprints as variable length arrays
            // later, so we add the nCols argument to Reader::_readSchema to allow SourceFitsReader
            // to call FitsReader::_readSchema in a way that prevents it from ever getting here.
            throw LSST_EXCEPT(
                afw::fits::FitsError,
                (boost::format("Unsupported FITS column type: '%s'.") % format).str()
            );
        }
    }

    // Add a field with a float or double element type to the schema.
    template <typename U>
    void addFloatField(Schema & schema, int size) const {
        if (size == 1) {
            if (cls == "Angle") {
                schema.addField< Angle >(name, doc, units);
            } else if (cls == "Array") {
                schema.addField< Array<U> >(name, doc, units, 1);
            } else if (cls == "Covariance") {
                schema.addField< Covariance<float> >(name, doc, units, 1);
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
                schema.addField< Covariance< Point<float> > >(name, doc, units);
                return;
            }
        } else if (size == 6) {
            if (cls == "Covariance(Moments)") {
                schema.addField< Covariance< Moments<float> > >(name, doc, units);
                return;
            }
        }
        if (cls == "Covariance") {
            double v = 0.5 * (std::sqrt(1 + 8 * size) - 1);
            int n = boost::math::iround(v);
            if (n * (n + 1) != size * 2) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    "Covariance field has invalid size."
                );
            }
            schema.addField< Covariance<float> >(name, doc, units, n);
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

} // anonymous

void FitsReader::_readSchema(
    Schema & schema,
    daf::base::PropertyList & metadata,
    bool stripMetadata
) {

    FitsSchema intermediate;
    int version = metadata.get("AFW_TABLE_VERSION", 0);

    int flagCol = metadata.get("FLAGCOL", 0);
    if (flagCol > 0) {
        metadata.remove("FLAGCOL");
        metadata.remove((boost::format("TTYPE%d") % flagCol).str());
        metadata.remove((boost::format("TFORM%d") % flagCol).str());
    }
    --flagCol; // switch from 1-indexed to 0-indexed

    try {
        std::vector<std::string> rawAliases = metadata.getArray<std::string>("ALIAS");
        for (std::vector<std::string>::const_iterator i = rawAliases.begin(); i != rawAliases.end(); ++i) {
            std::size_t pos = i->find_first_of(':');
            if (pos == std::string::npos) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    (boost::format("Malformed alias definition: '%s'") % (*i)).str()
                );
            }
            schema.getAliases()->set(i->substr(0, pos), i->substr(pos+1, std::string::npos));
        }
    } catch (pex::exceptions::NotFoundError &) {
        // if there are no aliases, just move on
    }
    metadata.remove("ALIAS");

    std::vector<std::string> keyList = metadata.getOrderedNames();
    for (std::vector<std::string>::const_iterator key = keyList.begin(); key != keyList.end(); ++key) {
        if (key->compare(0, 5, "TTYPE") == 0) {
            int col = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            std::string v = metadata.get<std::string>(*key);
            if (version < 1) {
                std::replace(v.begin(), v.end(), '_', '.');
            }
            intermediate.asColSet().modify(i, FitsSchema::SetName(v));
            if (i->doc.empty()) // don't overwrite if already set with TDOCn
                intermediate.asColSet().modify(i, FitsSchema::SetDoc(metadata.getComment(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TFLAG") == 0) {
            int bit = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::BitSet::iterator i = intermediate.asBitSet().lower_bound(bit);
            if (i == intermediate.asBitSet().end() || i->bit != bit) {
                i = intermediate.asBitSet().insert(i, FitsSchemaItem(-1, bit));
            }
            std::string v = metadata.get<std::string>(*key);
            if (version < 1) {
                std::replace(v.begin(), v.end(), '_', '.');
            }
            intermediate.asBitSet().modify(i, FitsSchema::SetName(v));
            if (i->doc.empty()) // don't overwrite if already set with TFDOCn
                intermediate.asBitSet().modify(i, FitsSchema::SetDoc(metadata.getComment(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 4, "TDOC") == 0) {
            int col = boost::lexical_cast<int>(key->substr(4)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            intermediate.asColSet().modify(i, FitsSchema::SetDoc(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TFDOC") == 0) {
            int bit = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::BitSet::iterator i = intermediate.asBitSet().lower_bound(bit);
            if (i == intermediate.asBitSet().end() || i->bit != bit) {
                i = intermediate.asBitSet().insert(i, FitsSchemaItem(-1, bit));
            }
            intermediate.asBitSet().modify(i, FitsSchema::SetDoc(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TUNIT") == 0) {
            int col = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            intermediate.asColSet().modify(i, FitsSchema::SetUnits(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TCCLS") == 0) {
            int col = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            intermediate.asColSet().modify(i, FitsSchema::SetCls(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TFORM") == 0) {
            int col = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            intermediate.asColSet().modify(i, FitsSchema::SetFormat(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        }
    }
    
    for (
        FitsSchema::List::const_iterator i = intermediate.asList().begin();
        i != intermediate.asList().end();
        ++i
    ) {
        if (i->bit >= 0) {
            schema.addField<Flag>(i->name, i->doc);
        } else if (i->col != flagCol) {
            i->addField(schema);
        }
    }
}

void FitsReader::_startRecords(BaseTable & table) {

    PTR(daf::base::PropertyList) metadata = table.getMetadata();
    // get the version number from the metadata.  If the entry is not there, set to 0
    // remove it from the metadata while the table is in memory
    int version = 0;
    if (metadata) {
        if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
        version = metadata->get<int>("AFW_TABLE_VERSION", 0);
        if (metadata->exists("AFW_TABLE_VERSION")) metadata->remove("AFW_TABLE_VERSION");
    }
    table.setVersion(version);

    _row = -1;
    _nRows = _fits->countRows();
    _processor = boost::make_shared<ProcessRecords>(_fits, _row);
    table.preallocate(_nRows);
}

PTR(BaseTable) FitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    Schema schema(*metadata, true);
    PTR(BaseTable) table = BaseTable::make(schema);
    table->setMetadata(metadata);
    _startRecords(*table);
    return table;
}

// ------------ FITS records reading ------------------------------------------------------------------------

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

    void operator()(SchemaItem<std::string> const & item) const {
        if (col == flagCol) ++col;
        std::string s;
        fits->readTableScalar(row, col, s);
        record->set(item.key, s);
        ++col;
    }

    void operator()(SchemaItem<Flag> const & item) const {
        assert(nFlags > 0);
        assert(flagCol >= 0);
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
        row(row_), col(0), bit(0), nFlags(0), flagCol(-1), fits(fits_)
    {
        fits->behavior &= ~ Fits::AUTO_CHECK; // temporarily disable automatic FITS exceptions
        fits->readKey("FLAGCOL", flagCol);
        if (fits->status == 0) {
            --flagCol; // we want 0-indexed column numbers, not FITS' 1-indexed numbers
            nFlags = fits->getTableArraySize(flagCol);
            if (nFlags) flags.reset(new bool[nFlags]);
        } else {
            fits->status = 0;
            flagCol = -1;
        }
        fits->behavior |= Fits::AUTO_CHECK;
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

PTR(FitsReader) FitsReader::make(Fits * fits, PTR(io::InputArchive) archive, int flags) {
    std::string name;
    fits->behavior &= ~Fits::AUTO_CHECK; // temporarily disable automatic FITS exceptions
    fits->readKey("AFW_TYPE", name);
    if (fits->status != 0) {
        name = "BASE";
        fits->status = 0;
    }
    fits->behavior |= Fits::AUTO_CHECK;
    Registry::iterator i = getRegistry().find(name);
    if (i == getRegistry().end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundError,
            (boost::format("FitsReader with name '%s' does not exist; check AFW_TYPE keyword.") % name).str()
        );
    }
    return (*i->second)(fits, archive, flags);
}

}}}} // namespace lsst::afw::table::io
