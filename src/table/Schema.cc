#include <list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/utils/hashCombine.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/io/FitsReader.h"
#include "lsst/afw/table/io/FitsSchemaInputMapper.h"
#include "lsst/afw/fits.h"

namespace lsst {
namespace afw {
namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- Miscellaneous Utilities -----------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

namespace {

inline char getDelimiter() { return '_'; }

// Concatenate two strings with a single-character delimiter between them
std::string join(std::string const &a, std::string const &b, char delimiter) {
    std::string full;
    full.reserve(a.size() + b.size() + 1);
    full += a;
    full.push_back(delimiter);
    full += b;
    return full;
}

// Functor to compare two ItemVariants for Key equality.
class ItemFunctors {
    using ItemVariant = detail::SchemaImpl::ItemVariant;

    // Compares keys (including types).
    struct KeyHelper {

        template <typename T>
        bool operator()(SchemaItem<T> const &a, SchemaItem<T> const &b) const {
            return a.key == b.key;
        }

        template <typename T, typename U>
        bool operator()(SchemaItem<T> const &a, SchemaItem<U> const &b) const {
            return false;
        }
    };

public:
    static bool compareKeys(ItemVariant const &a, ItemVariant const &b) {
        return std::visit(KeyHelper(), a, b);
    }

    static bool compareNames(ItemVariant const &a, ItemVariant const &b) {
        return std::visit(
            [](auto const & a, auto const & b) { return a.field.getName() == b.field.getName(); },
            a, b
        );
    }

    static bool compareDocs(ItemVariant const &a, ItemVariant const &b) {
        return std::visit(
            [](auto const & a, auto const & b) { return a.field.getDoc() == b.field.getDoc(); },
            a, b
        );
    }

    static bool compareUnits(ItemVariant const &a, ItemVariant const &b) {
        return std::visit(
            [](auto const & a, auto const & b) { return a.field.getUnits() == b.field.getUnits(); },
            a, b
        );
    }
};

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- SchemaImpl implementation ---------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

namespace detail {

template <typename T>
SchemaItem<T> SchemaImpl::find(std::string const &name) const {
    NameMap::const_iterator i = _names.lower_bound(name);
    if (i != _names.end() && i->first == name) {
        // got an exact match; we're done if it has the right type, and dead if it doesn't.
        try {
            return std::get<SchemaItem<T>>(_items[i->second]);
        } catch (std::bad_variant_access &err) {
            throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                                (boost::format("Field '%s' does not have the given type.") % name).str());
        }
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                      (boost::format("Field with name '%s' not found with type '%s'.") % name %
                       Field<T>::getTypeString())
                              .str());
}

template <typename T>
SchemaItem<T> SchemaImpl::find(Key<T> const &key) const {
    OffsetMap::const_iterator i = _offsets.lower_bound(key.getOffset());
    if (i != _offsets.end() && i->first == key.getOffset()) {
        try {
            return std::get<SchemaItem<T>>(_items[i->second]);
        } catch (std::bad_variant_access &err) {
            // just swallow the exception; this might be a subfield key that points to the beginning.
        }
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                      (boost::format("Field or subfield with offset %d not found with type '%s'.") %
                       key.getOffset() % Field<T>::getTypeString())
                              .str());
}

// We handle Flag fields separately when searching for keys, because their keys aren't like the others.
SchemaItem<Flag> SchemaImpl::find(Key<Flag> const &key) const {
    FlagMap::const_iterator i = _flags.lower_bound(std::make_pair(key.getOffset(), key.getBit()));
    if (i != _flags.end()) {
        if (i->first.first == key.getOffset() && i->first.second == key.getBit()) {
            try {
                return std::get<SchemaItem<Flag>>(_items[i->second]);
            } catch (std::bad_variant_access &err) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                                  (boost::format("Flag field with offset %d and bit %d not found.") %
                                   key.getOffset() % key.getBit())
                                          .str());
            }
        }
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                      (boost::format("Flag field with offset %d and bit %d not found.") % key.getOffset() %
                       key.getBit())
                              .str());
}

//----- Replacing an existing SchemaItem --------------------------------------------------------------------

// This is easier to understand if you start reading from the bottom of this section, with
// SchemaImpl::replaceField, then work your way up.

namespace {

// Find an exact SchemaItem by key ('exact' means no subfields, unlike the find member function above)
// Return the index into the item container.
template <typename T>
inline std::size_t findKey(SchemaImpl::OffsetMap const &offsets, SchemaImpl::FlagMap const &flags, Key<T> const &key,
                           bool throwIfMissing = true) {
    SchemaImpl::OffsetMap::const_iterator i = offsets.find(key.getOffset());
    if (i == offsets.end()) {
        if (throwIfMissing) {
            throw LSST_EXCEPT(pex::exceptions::NotFoundError,
                              (boost::format("Key of type %s with offset %d not found in Schema") %
                               Field<T>::getTypeString() % key.getOffset())
                                      .str());
        } else {
            return std::numeric_limits<size_t>::max();
        }
    }
    return i->second;
}

// Like the above, but special-cased for Flag
inline std::size_t findKey(SchemaImpl::OffsetMap const &offsets, SchemaImpl::FlagMap const &flags,
                           Key<Flag> const &key, bool throwIfMissing = true) {
    SchemaImpl::FlagMap::const_iterator i = flags.find(std::make_pair(key.getOffset(), key.getBit()));
    if (i == flags.end()) {
        if (throwIfMissing) {
            throw LSST_EXCEPT(
                    pex::exceptions::NotFoundError,
                    (boost::format("Key of type Flag with offset %d and bit %d not found in Schema") %
                     key.getOffset() % key.getBit())
                            .str());
        } else {
            return std::numeric_limits<size_t>::max();
        }
    }
    return i->second;
}

}  // namespace

template <typename T>
void SchemaImpl::replaceField(Key<T> const &key, Field<T> const &field) {
    NameMap::iterator j = _names.find(field.getName());
    SchemaItem<T> *item = nullptr;
    if (j != _names.end()) {
        // The field name is already present in the Schema; see if it's the one we're replacing.
        // If we can get the old item with this, we don't need to update the name map at all.
        item = std::get_if<SchemaItem<T>>(&_items[j->second]);
        if (!item || key != item->key) {
            throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterError,
                    (boost::format("Field with name '%s' already present in schema with a different key.") %
                     field.getName())
                            .str());
        }
    }
    if (!item) {  // Need to find the original item by key, since it's a new name.
        std::size_t index = findKey(_offsets, _flags, key);
        item = std::get_if<SchemaItem<T>>(&_items[index]);
        if (!item) {
            throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                              (boost::format("Incorrect key type '%s'.") % key).str());
        }
        j = _names.find(item->field.getName());
        _names.insert(j, std::pair<std::string, std::size_t>(field.getName(), j->second));
        _names.erase(j);
    }
    item->field = field;
}

//----- Other SchemaImpl things -----------------------------------------------------------------------------

template <typename T>
int SchemaImpl::contains(SchemaItem<T> const &item, int flags) const {
    if (!(flags & Schema::EQUAL_KEYS)) {
        throw LSST_EXCEPT(pex::exceptions::LogicError,
                          "Can only check whether item is in schema if flags & EQUAL_KEYS");
    }
    SchemaItem<T> const *cmpItem = nullptr;
    std::size_t index = findKey(_offsets, _flags, item.key, false);
    if (index != std::numeric_limits<size_t>::max()) {
        cmpItem = std::get_if<SchemaItem<T> >(&_items[index]);
        if (!cmpItem) {
            if ((flags & Schema::EQUAL_NAMES) && cmpItem->field.getName() != item.field.getName()) {
                flags &= ~Schema::EQUAL_NAMES;
            }
            if ((flags & Schema::EQUAL_DOCS) && cmpItem->field.getDoc() != item.field.getDoc()) {
                flags &= ~Schema::EQUAL_DOCS;
            }
            if ((flags & Schema::EQUAL_UNITS) && cmpItem->field.getUnits() != item.field.getUnits()) {
                flags &= ~Schema::EQUAL_UNITS;
            }
        }
    } else {
        flags = 0;
    }
    return flags;
}

std::set<std::string> SchemaImpl::getNames(bool topOnly) const {
    std::set<std::string> result;
    if (topOnly) {
        for (auto const &_name : _names) {
            std::size_t sep = _name.first.find(getDelimiter());
            if (sep == std::string::npos) {
                result.insert(result.end(), _name.first);
            } else {
                result.insert(result.end(), _name.first.substr(0, sep));
            }
        }
    } else {
        for (auto const &_name : _names) {
            result.insert(result.end(), _name.first);
        }
    }
    return result;
}

std::set<std::string> SchemaImpl::getNames(bool topOnly, std::string const &prefix) const {
    std::set<std::string> result;
    if (topOnly) {
        for (NameMap::const_iterator i = _names.lower_bound(prefix); i != _names.end(); ++i) {
            if (i->first.compare(0, prefix.size(), prefix) != 0) break;
            std::size_t sep = i->first.find(getDelimiter(), prefix.size() + 1);
            if (sep == std::string::npos) {
                result.insert(result.end(),
                              i->first.substr(prefix.size() + 1, i->first.size() - prefix.size()));
            } else {
                result.insert(result.end(), i->first.substr(prefix.size() + 1, sep - prefix.size() - 1));
            }
        }
    } else {
        for (NameMap::const_iterator i = _names.lower_bound(prefix); i != _names.end(); ++i) {
            if (i->first.compare(0, prefix.size(), prefix) != 0) break;
            result.insert(result.end(),
                          i->first.substr(prefix.size() + 1, i->first.size() - prefix.size() - 1));
        }
    }
    return result;
}

template <typename T>
Key<Array<T> > SchemaImpl::addField(Field<Array<T> > const &field, bool doReplace) {
    if (field.isVariableLength()) {
        // Variable-length array: allocate space for one ndarray
        return addFieldImpl(sizeof(ndarray::Array<T, 1, 1>), 1, field, doReplace);
    }
    // Fixed-length array: allocate space for getElementCount() elements of type T
    return addFieldImpl(sizeof(typename Field<T>::Element), field.getElementCount(), field, doReplace);
}

Key<std::string> SchemaImpl::addField(Field<std::string> const &field, bool doReplace) {
    if (field.isVariableLength()) {
        // Variable-length string: allocate space for one std::string
        return addFieldImpl(sizeof(std::string), 1, field, doReplace);
    }
    // Fixed-length string: allocate space for getElementCount() chars
    return addFieldImpl(sizeof(typename Field<std::string>::Element), field.getElementCount(), field,
                        doReplace);
}

template <typename T>
Key<T> SchemaImpl::addField(Field<T> const &field, bool doReplace) {
    return addFieldImpl(sizeof(typename Field<T>::Element), field.getElementCount(), field, doReplace);
}

Key<Flag> SchemaImpl::addField(Field<Flag> const &field, bool doReplace) {
    static std::size_t const ELEMENT_SIZE = sizeof(Field<Flag>::Element);
    std::pair<NameMap::iterator, bool> result =
            _names.insert(std::pair<std::string, std::size_t>(field.getName(), _items.size()));
    if (!result.second) {
        if (doReplace) {
            SchemaItem<Flag> *item = std::get_if<SchemaItem<Flag>>(&_items[result.first->second]);
            if (!item) {
                throw LSST_EXCEPT(
                        lsst::pex::exceptions::TypeError,
                        (boost::format("Cannot replace field with name '%s' because types differ.") %
                         field.getName())
                                .str());
            }
            if (item->field.getElementCount() != field.getElementCount()) {
                throw LSST_EXCEPT(
                        lsst::pex::exceptions::TypeError,
                        (boost::format("Cannot replace field with name '%s' because sizes differ.") %
                         field.getName())
                                .str());
            }
            item->field = field;
            return item->key;
        } else {
            throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterError,
                    (boost::format("Field with name '%s' already present in schema.") % field.getName())
                            .str());
        }
    } else {
        if (!_initFlag || _lastFlagBit >= ELEMENT_SIZE * 8) {
            std::size_t padding = ELEMENT_SIZE - _recordSize % ELEMENT_SIZE;
            if (padding != ELEMENT_SIZE) {
                _recordSize += padding;
            }
            _lastFlagField = _recordSize;
            _lastFlagBit = 0;
            _initFlag = true;
            _recordSize += field.getElementCount() * ELEMENT_SIZE;
        }
        SchemaItem<Flag> item(detail::Access::makeKey(_lastFlagField, _lastFlagBit), field);
        ++_lastFlagBit;
        _flags.insert(std::pair<std::pair<size_t, size_t>, size_t>(
                std::make_pair(item.key.getOffset(), item.key.getBit()), _items.size()));
        _items.push_back(item);
        return item.key;
    }
}

template <typename T>
Key<T> SchemaImpl::addFieldImpl(std::size_t elementSize, std::size_t elementCount, Field<T> const &field, bool doReplace) {
    std::pair<NameMap::iterator, bool> result =
            _names.insert(std::pair<std::string, std::size_t>(field.getName(), _items.size()));
    if (!result.second) {
        if (doReplace) {
            SchemaItem<T> *item = std::get_if<SchemaItem<T>>(&_items[result.first->second]);
            if (!item) {
                throw LSST_EXCEPT(
                        lsst::pex::exceptions::TypeError,
                        (boost::format("Cannot replace field with name '%s' because types differ.") %
                         field.getName())
                                .str());
            }
            // n.b. we don't use elementCount here because we *do* want variable length arrays (for
            // which we set elementCount == 1, but field->getElementCount() == -1) to compare as different
            // from fixed-length arrays with a single element.
            if (item->field.getElementCount() != field.getElementCount()) {
                throw LSST_EXCEPT(
                        lsst::pex::exceptions::TypeError,
                        (boost::format("Cannot replace field with name '%s' because sizes differ.") %
                         field.getName())
                                .str());
            }
            item->field = field;
            return item->key;
        } else {
            throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterError,
                    (boost::format("Field with name '%s' already present in schema.") % field.getName())
                            .str());
        }
    } else {
        std::size_t padding = elementSize - _recordSize % elementSize;
        if (padding != elementSize) {
            _recordSize += padding;
        }
        SchemaItem<T> item(detail::Access::makeKey(field, _recordSize), field);
        _recordSize += elementCount * elementSize;
        _offsets.insert(std::pair<std::size_t, std::size_t>(item.key.getOffset(), _items.size()));
        _items.push_back(item);
        return item.key;
    }
}

}  // namespace detail

//-----------------------------------------------------------------------------------------------------------
//----- Schema implementation -------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

int const Schema::VERSION;

Schema::Schema() : _impl(std::make_shared<Impl>()), _aliases(std::make_shared<AliasMap>()) {}

Schema::Schema(Schema const &other)  = default;
// Delegate to copy constructor  for backwards compatibility
Schema::Schema(Schema &&other) : Schema(other) {}

Schema &Schema::operator=(Schema const &) = default;
Schema &Schema::operator=(Schema &&) = default;
Schema::~Schema() = default;

Schema Schema::readFits(std::string const &filename, int hdu) {
    fits::Fits fp{filename, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK};
    fp.setHdu(hdu);
    return readFits(fp);
}

Schema Schema::readFits(fits::MemFileManager &manager, int hdu) {
    fits::Fits fp{manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK};
    fp.setHdu(hdu);
    return readFits(fp);
}

Schema Schema::readFits(fits::Fits &fitsfile) {
    daf::base::PropertyList header;
    fitsfile.readMetadata(header, false);
    return fromFitsMetadata(header);
}

Schema Schema::fromFitsMetadata(daf::base::PropertyList &header, bool stripMetadata) {
    return io::FitsSchemaInputMapper(header, stripMetadata).finalize();
}

std::string Schema::join(std::string const &a, std::string const &b) const {
    // delegate to utility funcs at top of this file
    return afw::table::join(a, b, getDelimiter());
}

void Schema::_edit() {
    if (!_impl.unique()) {
        std::shared_ptr<Impl> data(std::make_shared<Impl>(*_impl));
        _impl.swap(data);
    }
}

std::set<std::string> Schema::getNames(bool topOnly) const { return _impl->getNames(topOnly); }

template <typename T>
SchemaItem<T> Schema::find(std::string const &name) const {
    std::string tmp(name);
    _aliases->_apply(tmp);
    return _impl->find<T>(tmp);
}

template <typename T>
SchemaItem<T> Schema::find(Key<T> const &key) const {
    return _impl->find(key);
}

template <typename T>
Key<T> Schema::addField(Field<T> const &field, bool doReplace) {
    _edit();
    return _impl->addField(field, doReplace);
}

template <typename T>
void Schema::replaceField(Key<T> const &key, Field<T> const &field) {
    _edit();
    _impl->replaceField(key, field);
}

int Schema::contains(Schema const &other, int flags) const {
    if (_impl == other._impl) return flags;
    if (_impl->getItems().size() < other._impl->getItems().size()) return 0;
    int result = flags;
    if (result & EQUAL_FIELDS) {
        for (Impl::ItemContainer::const_iterator i1 = _impl->getItems().begin(),
                                                 i2 = other._impl->getItems().begin();
             i2 != other._impl->getItems().end(); ++i1, ++i2) {
            if ((result & EQUAL_KEYS) && !ItemFunctors::compareKeys(*i1, *i2)) result &= ~EQUAL_KEYS;
            if ((result & EQUAL_NAMES) && !ItemFunctors::compareNames(*i1, *i2)) result &= ~EQUAL_NAMES;
            if ((result & EQUAL_DOCS) && !ItemFunctors::compareDocs(*i1, *i2)) result &= ~EQUAL_DOCS;
            if ((result & EQUAL_UNITS) && !ItemFunctors::compareUnits(*i1, *i2)) result &= ~EQUAL_UNITS;
            if (!result) break;
        }
    }
    if ((result & EQUAL_ALIASES) && !getAliasMap()->contains(*other.getAliasMap())) result &= ~EQUAL_ALIASES;
    return result;
}

int Schema::compare(Schema const &other, int flags) const {
    int result = contains(other, flags);
    if (_impl->getItems().size() != other._impl->getItems().size()) {
        result &= ~EQUAL_FIELDS;
    }
    if (getAliasMap()->size() != other.getAliasMap()->size()) {
        result &= ~EQUAL_ALIASES;
    }
    return result;
}

std::size_t Schema::hash_value() const noexcept {
    // Completely arbitrary seed
    std::size_t result = 17;
    auto hasher = [&result](auto const &item) { result = utils::hashCombine(result, item.key); };
    forEach(hasher);
    return result;
}

template <typename T>
int Schema::contains(SchemaItem<T> const &item, int flags) const {
    return _impl->contains(item, flags);
}

void Schema::setAliasMap(std::shared_ptr<AliasMap> aliases) {
    if (!aliases) {
        aliases = std::make_shared<AliasMap>();
    }
    _aliases = aliases;
}

void Schema::disconnectAliases() { _aliases = std::make_shared<AliasMap>(*_aliases); }

//----- Stringification -------------------------------------------------------------------------------------

namespace {

// Schema::forEach functor used for stringificationx
struct Stream {
    using result_type = void;

    template <typename T>
    void operator()(SchemaItem<T> const &item) const {
        *os << "    (" << item.field << ", " << item.key << "),\n";
    }

    explicit Stream(std::ostream *os_) : os(os_) {}

    std::ostream *os;
};

}  // namespace

std::ostream &operator<<(std::ostream &os, Schema const &schema) {
    os << "Schema(\n";
    schema.forEach(Stream(&os));
    for (auto const &iter : *schema.getAliasMap()) {
        os << "    '" << iter.first << "'->'" << iter.second << "'\n";
    }
    return os << ")\n";
}

//-----------------------------------------------------------------------------------------------------------
//----- SubSchema implementation ----------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

std::string SubSchema::join(std::string const &a, std::string const &b) const {
    // delegate to utility funcs at top of this file
    return afw::table::join(a, b, getDelimiter());
}

SubSchema::SubSchema(std::shared_ptr<Impl> impl, std::shared_ptr<AliasMap> aliases, std::string const &name)
        : _impl(impl), _aliases(aliases), _name(name) {}

template <typename T>
SchemaItem<T> SubSchema::find(std::string const &name) const {
    return _impl->find<T>(_aliases->apply(join(_name, name)));
}

SubSchema SubSchema::operator[](std::string const &name) const {
    return SubSchema(_impl, _aliases, join(_name, name));
}

std::set<std::string> SubSchema::getNames(bool topOnly) const { return _impl->getNames(topOnly, _name); }

//-----------------------------------------------------------------------------------------------------------
//----- Explicit instantiation ------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// Note: by instantiating the public functions below, we also instantiate a lot of the private
// implementation functions.  If you move some of those to a different source file, you'll need
// more explicit instantiation.

#define INSTANTIATE_LAYOUT(r, data, elem)                                              \
    template Key<elem> Schema::addField(Field<elem> const &, bool);                    \
    template SchemaItem<elem> Schema::find(std::string const &) const;                 \
    template SchemaItem<elem> Schema::find(Key<elem> const &) const;                   \
    template SchemaItem<elem> detail::SchemaImpl::find(std::string const &name) const; \
    template int Schema::contains(SchemaItem<elem> const &, int) const;                \
    template void Schema::replaceField(Key<elem> const &, Field<elem> const &);        \
    template SchemaItem<elem> SubSchema::find(std::string const &) const;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_LAYOUT, _,
                      BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE))
}  // namespace table
}  // namespace afw
}  // namespace lsst
