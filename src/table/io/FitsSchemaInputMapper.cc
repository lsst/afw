// -*- lsst-c++ -*-

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>
#include <cctype>
#include <regex>

#include "boost/multi_index_container.hpp"
#include "boost/multi_index/sequenced_index.hpp"
#include "boost/multi_index/ordered_index.hpp"
#include "boost/multi_index/hashed_index.hpp"
#include "boost/multi_index/member.hpp"

#include "lsst/log/Log.h"
#include "lsst/geom.h"
#include "lsst/afw/table/io/FitsSchemaInputMapper.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

namespace {

// A quirk of Boost.MultiIndex (which we use for our container of FitsSchemaItems)
// that you have to use a special functor (like this one) to set data members
// in a container with set indices (because setting those values might require
// the element to be moved to a different place in the set).  Check out
// the Boost.MultiIndex docs for more information.
template <std::string FitsSchemaItem::*Member>
struct SetFitsSchemaString {
    void operator()(FitsSchemaItem &item) { item.*Member = _v; }
    explicit SetFitsSchemaString(std::string const &v) : _v(v) {}

private:
    std::string const &_v;
};

}  // namespace

class FitsSchemaInputMapper::Impl {
public:
    // A container class (based on Boost.MultiIndex) that provides three sort orders,
    // on column number, flag bit, and name (ttype).  This allows us to insert fields into the
    // schema in the correct order, regardless of which order they appear in the
    // FITS header.
    using InputContainer = boost::multi_index_container<FitsSchemaItem, boost::multi_index::indexed_by<boost::multi_index::ordered_non_unique<boost::multi_index::member<FitsSchemaItem, int, &FitsSchemaItem::column>>, boost::multi_index::ordered_non_unique<boost::multi_index::member<FitsSchemaItem, int, &FitsSchemaItem::bit>>, boost::multi_index::hashed_unique<boost::multi_index::member<FitsSchemaItem, std::string, &FitsSchemaItem::ttype>>, boost::multi_index::sequenced<>>>;

    // Typedefs for the special functors used to set data members.
    using SetTTYPE = SetFitsSchemaString<&FitsSchemaItem::ttype>;
    using SetTFORM = SetFitsSchemaString<&FitsSchemaItem::tform>;
    using SetTCCLS = SetFitsSchemaString<&FitsSchemaItem::tccls>;
    using SetTUNIT = SetFitsSchemaString<&FitsSchemaItem::tunit>;
    using SetDoc = SetFitsSchemaString<&FitsSchemaItem::doc>;

    // Typedefs for the different indices.
    using ByColumn = InputContainer::nth_index<0>::type;
    using ByBit = InputContainer::nth_index<1>::type;
    using ByName = InputContainer::nth_index<2>::type;
    using AsList = InputContainer::nth_index<3>::type;

    // Getters for the different indices.
    ByColumn &byColumn() { return inputs.get<0>(); }
    ByBit &byBit() { return inputs.get<1>(); }
    ByName &byName() { return inputs.get<2>(); }
    AsList &asList() { return inputs.get<3>(); }

    Impl()  {}

    int version{0};
    std::string type;
    int flagColumn{0};
    int archiveHdu{-1};
    Schema schema;
    std::vector<std::unique_ptr<FitsColumnReader>> readers;
    std::vector<Key<Flag>> flagKeys;
    std::unique_ptr<bool[]> flagWorkspace;
    std::shared_ptr<io::InputArchive> archive;
    InputContainer inputs;
    std::size_t nRowsToPrep = 1;
};

std::size_t FitsSchemaInputMapper::PREPPED_ROWS_FACTOR = 1 << 15;  // determined empirically; see DM-19461.

FitsSchemaInputMapper::FitsSchemaInputMapper(daf::base::PropertyList &metadata, bool stripMetadata)
        : _impl(std::make_shared<Impl>()) {
    // Set the table version.  If AFW_TABLE_VERSION tag exists, use that
    // If not, set to 0 if it has an AFW_TYPE, Schema default otherwise (DM-590)
    if (!metadata.exists("AFW_TYPE")) {
        _impl->version = lsst::afw::table::Schema::VERSION;
    }
    _impl->version = metadata.get("AFW_TABLE_VERSION", _impl->version);
    _impl->type = metadata.get("AFW_TYPE", _impl->type);
    if (stripMetadata) {
        metadata.remove("AFW_TABLE_VERSION");
    }
    if (stripMetadata) {
        metadata.remove("AFW_TYPE");
    }

    // Find a key that indicates an Archive stored on other HDUs
    _impl->archiveHdu = metadata.get("AR_HDU", -1);
    if (_impl->archiveHdu > 0) {
        --_impl->archiveHdu;  // AR_HDU is 1-indexed for historical reasons (RFC-304; see Source.cc)
        if (stripMetadata) {
            metadata.remove("AR_HDU");
        }
    }

    // Read aliases, stored as header entries with key 'ALIAS'
    try {
        std::vector<std::string> rawAliases = metadata.getArray<std::string>("ALIAS");
        for (auto const &rawAliase : rawAliases) {
            std::size_t pos = rawAliase.find_first_of(':');
            if (pos == std::string::npos) {
                throw LSST_EXCEPT(afw::fits::FitsError,
                                  (boost::format("Malformed alias definition: '%s'") % rawAliase).str());
            }
            _impl->schema.getAliasMap()->set(rawAliase.substr(0, pos), rawAliase.substr(pos + 1, std::string::npos));
        }
        if (stripMetadata) {
            metadata.remove("ALIAS");
        }
    } catch (pex::exceptions::NotFoundError &) {
        // if there are no aliases, just move on
    }

    if (_impl->version == 0) {
        // Read slots saved using an old mechanism in as aliases, since the new slot mechanism delegates
        // slot definition to the AliasMap.
        static std::array<std::pair<std::string, std::string>, 7> oldSlotKeys = {
                {std::make_pair("PSF_FLUX", "slot_PsfFlux"), std::make_pair("AP_FLUX", "slot_ApFlux"),
                 std::make_pair("INST_FLUX", "slot_GaussianFlux"),
                 std::make_pair("MODEL_FLUX", "slot_ModelFlux"),
                 std::make_pair("CALIB_FLUX", "slot_CalibFlux"), std::make_pair("CENTROID", "slot_Centroid"),
                 std::make_pair("SHAPE", "slot_Shape")}};
        for (auto const &oldSlotKey : oldSlotKeys) {
            std::string target = metadata.get(oldSlotKey.first + "_SLOT", std::string(""));
            if (!target.empty()) {
                _impl->schema.getAliasMap()->set(oldSlotKey.second, target);
                if (stripMetadata) {
                    metadata.remove(oldSlotKey.first);
                    metadata.remove(oldSlotKey.first + "_ERR_SLOT");
                    metadata.remove(oldSlotKey.first + "_FLAG_SLOT");
                }
            }
        }
    }

    // Read the rest of the header into the intermediate inputs container.
    std::vector<std::string> keyList = metadata.getOrderedNames();
    for (auto const &key : keyList) {
        if (key.compare(0, 5, "TTYPE") == 0) {
            int column = std::stoi(key.substr(5)) - 1;
            auto iter = _impl->byColumn().lower_bound(column);
            if (iter == _impl->byColumn().end() || iter->column != column) {
                iter = _impl->byColumn().insert(iter, FitsSchemaItem(column, -1));
            }
            std::string v = metadata.get<std::string>(key);
            _impl->byColumn().modify(iter, Impl::SetTTYPE(v));
            if (iter->doc.empty()) {  // don't overwrite if already set with TDOCn
                _impl->byColumn().modify(iter, Impl::SetDoc(metadata.getComment(key)));
            }
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TFLAG") == 0) {
            int bit = std::stoi(key.substr(5)) - 1;
            auto iter = _impl->byBit().lower_bound(bit);
            if (iter == _impl->byBit().end() || iter->bit != bit) {
                iter = _impl->byBit().insert(iter, FitsSchemaItem(-1, bit));
            }
            std::string v = metadata.get<std::string>(key);
            _impl->byBit().modify(iter, Impl::SetTTYPE(v));
            if (iter->doc.empty()) {  // don't overwrite if already set with TFDOCn
                _impl->byBit().modify(iter, Impl::SetDoc(metadata.getComment(key)));
            }
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 4, "TDOC") == 0) {
            int column = std::stoi(key.substr(4)) - 1;
            auto iter = _impl->byColumn().lower_bound(column);
            if (iter == _impl->byColumn().end() || iter->column != column) {
                iter = _impl->byColumn().insert(iter, FitsSchemaItem(column, -1));
            }
            _impl->byColumn().modify(iter, Impl::SetDoc(metadata.get<std::string>(key)));
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TFDOC") == 0) {
            int bit = std::stoi(key.substr(5)) - 1;
            auto iter = _impl->byBit().lower_bound(bit);
            if (iter == _impl->byBit().end() || iter->bit != bit) {
                iter = _impl->byBit().insert(iter, FitsSchemaItem(-1, bit));
            }
            _impl->byBit().modify(iter, Impl::SetDoc(metadata.get<std::string>(key)));
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TUNIT") == 0) {
            int column = std::stoi(key.substr(5)) - 1;
            auto iter = _impl->byColumn().lower_bound(column);
            if (iter == _impl->byColumn().end() || iter->column != column) {
                iter = _impl->byColumn().insert(iter, FitsSchemaItem(column, -1));
            }
            _impl->byColumn().modify(iter, Impl::SetTUNIT(metadata.get<std::string>(key)));
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TCCLS") == 0) {
            int column = std::stoi(key.substr(5)) - 1;
            auto iter = _impl->byColumn().lower_bound(column);
            if (iter == _impl->byColumn().end() || iter->column != column) {
                iter = _impl->byColumn().insert(iter, FitsSchemaItem(column, -1));
            }
            _impl->byColumn().modify(iter, Impl::SetTCCLS(metadata.get<std::string>(key)));
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TFORM") == 0) {
            int column = std::stoi(key.substr(5)) - 1;
            auto iter = _impl->byColumn().lower_bound(column);
            if (iter == _impl->byColumn().end() || iter->column != column) {
                iter = _impl->byColumn().insert(iter, FitsSchemaItem(column, -1));
            }
            _impl->byColumn().modify(iter, Impl::SetTFORM(metadata.get<std::string>(key)));
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TZERO") == 0) {
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TSCAL") == 0) {
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TNULL") == 0) {
            if (stripMetadata) {
                metadata.remove(key);
            }
        } else if (key.compare(0, 5, "TDISP") == 0) {
            if (stripMetadata) {
                metadata.remove(key);
            }
        }
    }

    // Find the column used to store flags, and setup the flag-handling data members from it.
    _impl->flagColumn = metadata.get("FLAGCOL", 0);
    if (_impl->flagColumn > 0) {
        if (stripMetadata) {
            metadata.remove("FLAGCOL");
        }
        --_impl->flagColumn;  // switch from 1-indexed to 0-indexed
        auto iter = _impl->byColumn().find(_impl->flagColumn);
        if (iter == _impl->byColumn().end()) {
            throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    (boost::format("Column for flag data not found; FLAGCOL=%d") % _impl->flagColumn).str());
        }
        // Regex to unpack a FITS TFORM value for a bit array column (TFORM code 'X').  The number
        // that precedes the code is the size of the array; the number that follows it (if present)
        // is ignored.
        static std::regex const regex("(\\d+)?X\\(?(\\d)*\\)?");
        std::smatch m;
        if (!std::regex_match(iter->tform, m, regex)) {
            throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    (boost::format("Invalid TFORM key for flags column: '%s'") % iter->tform).str());
        }
        int nFlags = 1;
        if (m[1].matched) {
            nFlags = std::stoi(m[1].str());
        }
        _impl->flagKeys.resize(nFlags);
        _impl->flagWorkspace.reset(new bool[nFlags]);
        // Delete the flag column from the input list so we don't interpret it as a
        // regular field.
        _impl->byColumn().erase(iter);
    }
}

FitsSchemaInputMapper::FitsSchemaInputMapper(FitsSchemaInputMapper const &) = default;
FitsSchemaInputMapper::FitsSchemaInputMapper(FitsSchemaInputMapper &&) = default;
FitsSchemaInputMapper &FitsSchemaInputMapper::operator=(FitsSchemaInputMapper const &) = default;
FitsSchemaInputMapper &FitsSchemaInputMapper::operator=(FitsSchemaInputMapper &&) = default;
FitsSchemaInputMapper::~FitsSchemaInputMapper() = default;

void FitsSchemaInputMapper::setArchive(std::shared_ptr<InputArchive> archive) { _impl->archive = archive; }

bool FitsSchemaInputMapper::readArchive(afw::fits::Fits &fits) {
    int oldHdu = fits.getHdu();
    if (_impl->archiveHdu < 0) _impl->archiveHdu = oldHdu + 1;
    try {
        fits.setHdu(_impl->archiveHdu);
        _impl->archive.reset(new io::InputArchive(InputArchive::readFits(fits)));
        fits.setHdu(oldHdu);
        return true;
    } catch (afw::fits::FitsError &) {
        fits.status = 0;
        fits.setHdu(oldHdu);
        _impl->archiveHdu = -1;
        return false;
    }
}

bool FitsSchemaInputMapper::hasArchive() const { return static_cast<bool>(_impl->archive); }

FitsSchemaItem const *FitsSchemaInputMapper::find(std::string const &ttype) const {
    auto iter = _impl->byName().find(ttype);
    if (iter == _impl->byName().end()) {
        return nullptr;
    }
    return &(*iter);
}

FitsSchemaItem const *FitsSchemaInputMapper::find(int column) const {
    auto iter = _impl->byColumn().lower_bound(column);
    if (iter == _impl->byColumn().end() || iter->column != column) {
        return nullptr;
    }
    return &(*iter);
}

void FitsSchemaInputMapper::erase(Item const *item) {
    auto iter = _impl->byColumn().lower_bound(item->column);
    assert(iter != _impl->byColumn().end() && iter->column == item->column);
    _impl->byColumn().erase(iter);
}

void FitsSchemaInputMapper::erase(std::string const &ttype) {
    auto iter = _impl->byName().find(ttype);
    if (iter != _impl->byName().end() && iter->ttype == ttype) {
        _impl->byName().erase(iter);
    }
}

void FitsSchemaInputMapper::erase(int column) {
    auto iter = _impl->byColumn().lower_bound(column);
    if (iter != _impl->byColumn().end() && iter->column == column) {
        _impl->byColumn().erase(iter);
    }
}

void erase(int column);

void FitsSchemaInputMapper::customize(std::unique_ptr<FitsColumnReader> reader) {
    _impl->readers.push_back(std::move(reader));
}

namespace {

template <typename T>
class StandardReader : public FitsColumnReader {
public:
    static std::unique_ptr<FitsColumnReader> make(Schema &schema, FitsSchemaItem const &item,
                                                  FieldBase<T> const &base = FieldBase<T>()) {
        return std::unique_ptr<FitsColumnReader>(new StandardReader(schema, item, base));
    }

    StandardReader(Schema &schema, FitsSchemaItem const &item, FieldBase<T> const &base)
            : _column(item.column), _key(schema.addField<T>(item.ttype, item.doc, item.tunit, base)),
              _cache(), _cacheFirstRow(0)
    {}

    void prepRead(std::size_t firstRow, std::size_t nRows, fits::Fits & fits) override {
        // We only prep and cache scalar-valued columns, not array-valued
        // columns, as apparently the order CFITSIO reads array-valued columns
        // is not the order we want.
        if (_key.getElementCount() == 1u) {
            std::size_t nElements = nRows*_key.getElementCount();
            _cache.resize(nElements);
            _cacheFirstRow = firstRow;
            fits.readTableArray(firstRow, _column, nElements, &_cache.front());
        }
    }

    void readCell(BaseRecord &record, std::size_t row, afw::fits::Fits &fits,
                  std::shared_ptr<InputArchive> const &archive) const override {
        if (_cache.empty()) {
            fits.readTableArray(row, _column, _key.getElementCount(), record.getElement(_key));
        } else {
            assert(row >= _cacheFirstRow);
            std::size_t offset = row - _cacheFirstRow;
            assert(offset < _cache.size());
            std::copy_n(_cache.begin() + offset, _key.getElementCount(), record.getElement(_key));
        }
    }

private:
    int _column;
    Key<T> _key;
    std::vector<typename FieldBase<T>::Element> _cache;
    std::size_t _cacheFirstRow;
    std::size_t _nRowsToPrep;
};

class AngleReader : public FitsColumnReader {
public:
    static std::unique_ptr<FitsColumnReader> make(
            Schema &schema, FitsSchemaItem const &item,
            FieldBase<lsst::geom::Angle> const &base = FieldBase<lsst::geom::Angle>()) {
        return std::unique_ptr<FitsColumnReader>(new AngleReader(schema, item, base));
    }

    AngleReader(Schema &schema, FitsSchemaItem const &item, FieldBase<lsst::geom::Angle> const &base)
            : _column(item.column), _key(schema.addField<lsst::geom::Angle>(item.ttype, item.doc, "", base)) {
        // We require an LSST-specific key in the headers before parsing a column
        // as Angle at all, so we don't need to worry about other units or other
        // spellings of radians.  We do continue to support no units for backwards
        // compatibility.
        if (!item.tunit.empty() && item.tunit != "rad") {
            throw LSST_EXCEPT(afw::fits::FitsError,
                              "Angle fields must be persisted in radians (TUNIT='rad').");
        }
    }

    void prepRead(std::size_t firstRow, std::size_t nRows, fits::Fits & fits) override {
        assert(_key.getElementCount() == 1u);
        _cache.resize(nRows);
        _cacheFirstRow = firstRow;
        fits.readTableArray(firstRow, _column, nRows, &_cache.front());
    }

    void readCell(BaseRecord &record, std::size_t row, afw::fits::Fits &fits,
                  std::shared_ptr<InputArchive> const &archive) const override {
        if (_cache.empty()) {
            double tmp = 0;
            fits.readTableScalar(row, _column, tmp);
            record.set(_key, tmp * lsst::geom::radians);
        } else {
            assert(row >= _cacheFirstRow);
            std::size_t offset = row - _cacheFirstRow;
            assert(offset < _cache.size());
            record.set(_key, _cache[offset] * lsst::geom::radians);
        }
    }

private:
    int _column;
    Key<lsst::geom::Angle> _key;
    std::vector<double> _cache;
    std::size_t _cacheFirstRow;
};

class StringReader : public FitsColumnReader {
public:
    static std::unique_ptr<FitsColumnReader> make(Schema &schema, FitsSchemaItem const &item, int size) {
        return std::unique_ptr<FitsColumnReader>(new StringReader(schema, item, size));
    }

    StringReader(Schema &schema, FitsSchemaItem const &item, int size)
            : _column(item.column),
              _key(schema.addField<std::string>(item.ttype, item.doc, item.tunit, size)),
              _isVariableLength(size == 0) {}

    void readCell(BaseRecord &record, std::size_t row, afw::fits::Fits &fits,
                  std::shared_ptr<InputArchive> const &archive) const override {
        std::string s;
        fits.readTableScalar(row, _column, s, _isVariableLength);
        record.set(_key, s);
    }

private:
    int _column;
    Key<std::string> _key;
    bool _isVariableLength;
};

template <typename T>
class VariableLengthArrayReader : public FitsColumnReader {
public:
    static std::unique_ptr<FitsColumnReader> make(Schema &schema, FitsSchemaItem const &item) {
        return std::unique_ptr<FitsColumnReader>(new VariableLengthArrayReader(schema, item));
    }

    VariableLengthArrayReader(Schema &schema, FitsSchemaItem const &item)
            : _column(item.column), _key(schema.addField<Array<T>>(item.ttype, item.doc, item.tunit, 0)) {}

    void readCell(BaseRecord &record, std::size_t row, afw::fits::Fits &fits,
                  std::shared_ptr<InputArchive> const &archive) const override {
        int size = fits.getTableArraySize(row, _column);
        ndarray::Array<T, 1, 1> array = ndarray::allocate(size);
        fits.readTableArray(row, _column, size, array.getData());
        record.set(_key, array);
    }

private:
    int _column;
    Key<Array<T>> _key;
};

// Read a 2-element FITS array column as separate x and y Schema fields (hence converting
// from the old Point compound field to the new PointKey FunctorKey).
template <typename T>
class PointConversionReader : public FitsColumnReader {
public:
    static std::unique_ptr<FitsColumnReader> make(Schema &schema, FitsSchemaItem const &item) {
        return std::unique_ptr<FitsColumnReader>(new PointConversionReader(schema, item));
    }

    PointConversionReader(Schema &schema, FitsSchemaItem const &item)
            : _column(item.column), _key(PointKey<T>::addFields(schema, item.ttype, item.doc, item.tunit)) {}

    void readCell(BaseRecord &record, std::size_t row, afw::fits::Fits &fits,
                  std::shared_ptr<InputArchive> const &archive) const override {
        std::array<T, 2> buffer;
        fits.readTableArray(row, _column, 2, buffer.data());
        record.set(_key, lsst::geom::Point<T, 2>(buffer[0], buffer[1]));
    }

private:
    int _column;
    PointKey<T> _key;
};

// Read a 2-element FITS array column as separate ra and dec Schema fields (hence converting
// from the old Coord compound field to the new CoordKey FunctorKey).
class CoordConversionReader : public FitsColumnReader {
public:
    static std::unique_ptr<FitsColumnReader> make(Schema &schema, FitsSchemaItem const &item) {
        return std::unique_ptr<FitsColumnReader>(new CoordConversionReader(schema, item));
    }

    CoordConversionReader(Schema &schema, FitsSchemaItem const &item)
            : _column(item.column), _key(CoordKey::addFields(schema, item.ttype, item.doc)) {}

    void readCell(BaseRecord &record, std::size_t row, afw::fits::Fits &fits,
                  std::shared_ptr<InputArchive> const &archive) const override {
        std::array<lsst::geom::Angle, 2> buffer;
        fits.readTableArray(row, _column, 2, buffer.data());
        record.set(_key, lsst::geom::SpherePoint(buffer[0], buffer[1]));
    }

private:
    int _column;
    CoordKey _key;
};

// Read a 3-element FITS array column as separate xx, yy, and xy Schema fields (hence converting
// from the old Moments compound field to the new QuadrupoleKey FunctorKey).
class MomentsConversionReader : public FitsColumnReader {
public:
    static std::unique_ptr<FitsColumnReader> make(Schema &schema, FitsSchemaItem const &item) {
        return std::unique_ptr<FitsColumnReader>(new MomentsConversionReader(schema, item));
    }

    MomentsConversionReader(Schema &schema, FitsSchemaItem const &item)
            : _column(item.column),
              _key(QuadrupoleKey::addFields(schema, item.ttype, item.doc, CoordinateType::PIXEL)) {}

    void readCell(BaseRecord &record, std::size_t row, afw::fits::Fits &fits,
                  std::shared_ptr<InputArchive> const &archive) const override {
        std::array<double, 3> buffer;
        fits.readTableArray(row, _column, 3, buffer.data());
        record.set(_key, geom::ellipses::Quadrupole(buffer[0], buffer[1], buffer[2], false));
    }

private:
    int _column;
    QuadrupoleKey _key;
};

// Read a FITS array column representing a packed symmetric matrix into
// Schema fields for each element (hence converting from the old Covariance
// compound field to the new CovarianceMatrixKey FunctorKey).
template <typename T, int N>
class CovarianceConversionReader : public FitsColumnReader {
public:
    static std::string guessUnits(std::string const &oldUnits) {
        static std::regex const regex("(.*)(\\^(\\d+))?");
        std::smatch m;
        if (!std::regex_match(oldUnits, m, regex)) {
            int oldPower = std::stoi(m[2]);
            int newPower = std::sqrt(oldPower);
            return std::to_string(newPower);
        }
        return oldUnits;
    }

    static std::unique_ptr<FitsColumnReader> make(Schema &schema, FitsSchemaItem const &item,
                                                  std::vector<std::string> const &names) {
        return std::unique_ptr<FitsColumnReader>(new CovarianceConversionReader(schema, item, names));
    }

    CovarianceConversionReader(Schema &schema, FitsSchemaItem const &item,
                               std::vector<std::string> const &names)
            : _column(item.column),
              _size(names.size()),
              _key(CovarianceMatrixKey<T, N>::addFields(schema, item.ttype, names, guessUnits(item.tunit))),
              _buffer(new T[detail::computeCovariancePackedSize(names.size())]) {}

    void readCell(BaseRecord &record, std::size_t row, afw::fits::Fits &fits,
                  std::shared_ptr<InputArchive> const &archive) const override {
        fits.readTableArray(row, _column, detail::computeCovariancePackedSize(_size), _buffer.get());
        for (int i = 0; i < _size; ++i) {
            for (int j = i; j < _size; ++j) {
                _key.setElement(record, i, j, _buffer[detail::indexCovariance(i, j)]);
            }
        }
    }

private:
    int _column;
    int _size;
    CovarianceMatrixKey<T, N> _key;
    std::unique_ptr<T[]> _buffer;
};

std::unique_ptr<FitsColumnReader> makeColumnReader(Schema &schema, FitsSchemaItem const &item) {
    // Regex to unpack a FITS TFORM value.  The first number is the size of the array (1 if not present),
    // followed by an alpha code indicating the type (preceded by P or Q for variable size array).
    // The last number is ignored.
    static std::regex const regex("(\\d+)?([PQ])?([A-Z])\\(?(\\d)*\\)?");
    // start by parsing the format; this tells the element type of the field and the number of elements
    std::smatch m;
    if (!std::regex_match(item.tform, m, regex)) {
        return std::unique_ptr<FitsColumnReader>();
    }
    int size = 1;
    if (m[1].matched) {
        size = std::stoi(m[1].str());
    }
    char code = m[3].str()[0];
    if (m[2].matched) {
        // P or Q presence indicates a variable-length array, which we can get by just setting the
        // size to zero and letting the rest of the logic run its course.
        size = 0;
    }
    // switch code over FITS codes that correspond to different element types
    switch (code) {
        case 'B':  // 8-bit unsigned integers -- can only be scalars or Arrays
            if (size == 1) {
                if (item.tccls == "Array") {
                    return StandardReader<Array<std::uint8_t>>::make(schema, item, size);
                }
                return StandardReader<std::uint8_t>::make(schema, item);
            }
            if (size == 0) {
                return VariableLengthArrayReader<std::uint8_t>::make(schema, item);
            }
            return StandardReader<Array<std::uint8_t>>::make(schema, item, size);

        case 'I':  // 16-bit integers - can only be scalars or Arrays (we assume they're unsigned, since
                   // that's all we ever write, and CFITSIO will complain later if they aren't)
            if (size == 1) {
                if (item.tccls == "Array") {
                    return StandardReader<Array<std::uint16_t>>::make(schema, item, size);
                }
                return StandardReader<std::uint16_t>::make(schema, item);
            }
            if (size == 0) {
                return VariableLengthArrayReader<std::uint16_t>::make(schema, item);
            }
            return StandardReader<Array<std::uint16_t>>::make(schema, item, size);
        case 'J':  // 32-bit integers - can only be scalars, Point fields, or Arrays
            if (size == 0) {
                return VariableLengthArrayReader<std::int32_t>::make(schema, item);
            }
            if (item.tccls == "Point") {
                return PointConversionReader<std::int32_t>::make(schema, item);
            }
            if (size > 1 || item.tccls == "Array") {
                return StandardReader<Array<std::int32_t>>::make(schema, item, size);
            }
            return StandardReader<std::int32_t>::make(schema, item);
        case 'K':  // 64-bit integers - can only be scalars.
            if (size == 1) {
                return StandardReader<std::int64_t>::make(schema, item);
            }
        case 'E':  // floats
            if (size == 0) {
                return VariableLengthArrayReader<float>::make(schema, item);
            }
            if (size == 1) {
                if (item.tccls == "Array") {
                    return StandardReader<Array<float>>::make(schema, item, 1);
                }
                // Just use scalars for Covariances of size 1, since that results in more
                // natural field names (essentially never happens anyway).
                return StandardReader<float>::make(schema, item);
            }
            if (size == 3 && item.tccls == "Covariance(Point)") {
                std::vector<std::string> names = {"x", "y"};
                return CovarianceConversionReader<float, 2>::make(schema, item, names);
            }
            if (size == 6 && item.tccls == "Covariance(Moments)") {
                std::vector<std::string> names = {"xx", "yy", "xy"};
                return CovarianceConversionReader<float, 3>::make(schema, item, names);
            }
            if (item.tccls == "Covariance") {
                double v = 0.5 * (std::sqrt(1 + 8 * size) - 1);
                std::size_t n = std::lround(v);
                if (n * (n + 1) != size * 2) {
                    throw LSST_EXCEPT(afw::fits::FitsError, "Covariance field has invalid size.");
                }
                std::vector<std::string> names(n);
                for (std::size_t i = 0; i < n; ++i) {
                    names[i] = std::to_string(i);
                }
                return CovarianceConversionReader<float, Eigen::Dynamic>::make(schema, item, names);
            }
            return StandardReader<Array<float>>::make(schema, item, size);
        case 'D':  // doubles
            if (size == 0) {
                return VariableLengthArrayReader<double>::make(schema, item);
            }
            if (size == 1) {
                if (item.tccls == "Angle") {
                    return AngleReader::make(schema, item);
                }
                if (item.tccls == "Array") {
                    return StandardReader<Array<double>>::make(schema, item, 1);
                }
                return StandardReader<double>::make(schema, item);
            }
            if (size == 2) {
                if (item.tccls == "Point") {
                    return PointConversionReader<double>::make(schema, item);
                }
                if (item.tccls == "Coord") {
                    return CoordConversionReader::make(schema, item);
                }
            }
            if (size == 3 && item.tccls == "Moments") {
                return MomentsConversionReader::make(schema, item);
            }
            return StandardReader<Array<double>>::make(schema, item, size);
        case 'A':  // strings
            // StringReader can read both fixed-length and variable-length (size=0) strings
            return StringReader::make(schema, item, size);
        default:
            return std::unique_ptr<FitsColumnReader>();
    }
}

bool endswith(std::string const &s, std::string const &suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool isInstFlux(FitsSchemaItem const & item) {
    // helper lambda to make reading the real logic easier
    auto includes = [](std::string const & s, char const * target) {
        return s.find(target) != std::string::npos;
    };
    if (!includes(item.ttype, "flux")) return false;
    if (includes(item.ttype, "modelfit_CModel") && item.tunit.empty()) {
        // CModel flux fields were written with no units prior to DM-16068,
        // but should have been "count".
        return true;
    }
    // transform units to lowercase.
    std::string units(item.tunit);
    std::transform(units.begin(), units.end(), units.begin(), [](char c) { return std::tolower(c); } );
    return includes(units, "count") || includes(units, "dn") || includes (units, "adu");
}

// Replace 'from' with 'to' in 'full', returning the result.
std::string replace(std::string full, std::string const & from, std::string const & to) {
    return full.replace(full.find(from), from.size(), to);
}

}  // namespace

Schema FitsSchemaInputMapper::finalize() {
    if (_impl->version == 0) {
        AliasMap &aliases = *_impl->schema.getAliasMap();
        for (auto iter = _impl->asList().begin(); iter != _impl->asList().end(); ++iter) {
            std::size_t flagPos = iter->ttype.find("flags");
            if (flagPos != std::string::npos) {
                // We want to create aliases that resolve "(.*)_flag" to "$1_flags"; old schemas will have
                // the latter, but new conventions (including slots) expect the former.
                // But we can't do that, because adding that alias directly results in a cycle in the
                // aliases (since aliases do partial matches, and keep trying until there are no matches,
                // we'd have "(.*)_flag" resolve to "$1_flagssssssssssssss...").
                // Instead, we *rename* from "flags" to "flag", then create the reverse alias.
                std::string ttype = iter->ttype;
                std::string prefix = iter->ttype.substr(0, flagPos);
                ttype.replace(flagPos, 5, "flag");
                _impl->asList().modify(iter, Impl::SetTTYPE(ttype));
                // Note that we're not aliasing the full field, just the first part - if we have multiple
                // flag fields, one alias should be sufficient for all of them (because of partial matching).
                // Of course, we'll try to recreate that alias every time we handle another flag field with
                // the same prefix, but AliasMap know hows to handle that no-op set.
                aliases.set(prefix + "flags", prefix + "flag");
            } else if (isInstFlux(*iter)) {
                // Create an alias that resolves "X_instFlux" to "X" or "X_instFluxErr" to "X_err".
                if (endswith(iter->ttype, "_err")) {
                    aliases.set(replace(iter->ttype, "_err", "_instFluxErr"), iter->ttype);
                } else {
                    aliases.set(iter->ttype + "_instFlux", iter->ttype);
                }
            } else if (endswith(iter->ttype, "_err")) {
                // Create aliases that resolve "(.*)_(.*)Err" and "(.*)_(.*)_(.*)_Cov" to
                // "$1_err_$2Err" and "$1_err_$2_$3_Cov", to make centroid and shape uncertainties
                // available under the new conventions.  We don't have to create aliases for the
                // centroid and shape values themselves, as those will automatically be correct
                // after the PointConversionReader and MomentsConversionReader do their work.
                if (iter->tccls == "Covariance(Point)") {
                    aliases.set(replace(iter->ttype, "_err", "_yErr"), iter->ttype + "_yErr");
                    aliases.set(replace(iter->ttype, "_err", "_xErr"), iter->ttype + "_xErr");
                    aliases.set(replace(iter->ttype, "_err", "_x_y_Cov"), iter->ttype + "_x_y_Cov");
                } else if (iter->tccls == "Covariance(Moments)") {
                    aliases.set(replace(iter->ttype, "_err", "_xxErr"), iter->ttype + "_xxErr");
                    aliases.set(replace(iter->ttype, "_err", "_yyErr"), iter->ttype + "_yyErr");
                    aliases.set(replace(iter->ttype, "_err", "_xyErr"), iter->ttype + "_xyErr");
                    aliases.set(replace(iter->ttype, "_err", "_xx_yy_Cov"), iter->ttype + "_xx_yy_Cov");
                    aliases.set(replace(iter->ttype, "_err", "_xx_xy_Cov"), iter->ttype + "_xx_xy_Cov");
                    aliases.set(replace(iter->ttype, "_err", "_yy_xy_Cov"), iter->ttype + "_yy_xy_Cov");
                }
            }
        }
    } else if (_impl->version < 3) {
        // Version == 1 tables use Sigma when we should use Err (see RFC-333) and had no fields
        // that should have been named Sigma. So provide aliases xErr -> xSigma.
        // Version <= 2 tables used _flux when we should use _instFlux (see RFC-322).
        AliasMap &aliases = *_impl->schema.getAliasMap();
        for (auto iter = _impl->asList().begin(); iter != _impl->asList().end(); ++iter) {
            std::string name = iter->ttype;
            if (_impl->version < 2 && endswith(name, "Sigma")) {
                name = replace(std::move(name), "Sigma", "Err");
            }
            if (_impl->version < 3 && isInstFlux(*iter)) {
                name = replace(std::move(name), "flux", "instFlux");
            }
            if (name != iter->ttype) {
                aliases.set(name, iter->ttype);
            }
        }
    }
    for (auto iter = _impl->asList().begin(); iter != _impl->asList().end(); ++iter) {
        if (iter->bit < 0) {  // not a Flag column
            std::unique_ptr<FitsColumnReader> reader = makeColumnReader(_impl->schema, *iter);
            if (reader) {
                _impl->readers.push_back(std::move(reader));
            } else {
                LOGLS_WARN("lsst.afw.FitsSchemaInputMapper", "Format " << iter->tform << " for column "
                                                                  << iter->ttype
                                                                  << " not supported; skipping.");
            }
        } else {  // is a Flag column
            if (static_cast<std::size_t>(iter->bit) >= _impl->flagKeys.size()) {
                throw LSST_EXCEPT(afw::fits::FitsError,
                                  (boost::format("Flag field '%s' is is in bit %d (0-indexed) of only %d") %
                                   iter->ttype % iter->bit % _impl->flagKeys.size())
                                          .str());
            }
            _impl->flagKeys[iter->bit] = _impl->schema.addField<Flag>(iter->ttype, iter->doc);
        }
    }
    _impl->asList().clear();
    if (_impl->schema.getRecordSize() <= 0) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            (boost::format("Non-positive record size: %d; file is corrupt or invalid.") %
            _impl->schema.getRecordSize()).str()
        );
    }
    _impl->nRowsToPrep = std::max(PREPPED_ROWS_FACTOR / _impl->schema.getRecordSize(), std::size_t(1));
    return _impl->schema;
}

void FitsSchemaInputMapper::readRecord(BaseRecord &record, afw::fits::Fits &fits, std::size_t row) {
    if (!_impl->flagKeys.empty()) {
        fits.readTableArray<bool>(row, _impl->flagColumn, _impl->flagKeys.size(), _impl->flagWorkspace.get());
        for (std::size_t bit = 0; bit < _impl->flagKeys.size(); ++bit) {
            record.set(_impl->flagKeys[bit], _impl->flagWorkspace[bit]);
        }
    }
    if (_impl->nRowsToPrep != 1 && row % _impl->nRowsToPrep == 0) {
        // Give readers a chance to read and cache up to nRowsToPrep rows-
        // worth of values.
        std::size_t size = std::min(_impl->nRowsToPrep, fits.countRows() - row);
        for (auto const &reader : _impl->readers) {
            reader->prepRead(row, size, fits);
        }
    }
    for (auto const & reader : _impl->readers) {
        reader->readCell(record, row, fits, _impl->archive);
    }
}
}  // namespace io
}  // namespace table
}  // namespace afw
}  // namespace lsst
