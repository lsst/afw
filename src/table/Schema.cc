#include <list>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "boost/mpl/and.hpp"
#include "boost/mpl/bool.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/iterator/filter_iterator.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

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
    typedef detail::SchemaImpl::ItemVariant ItemVariant;

    // Compares keys - must be initialized with one ItemVariant and passed the other.
    struct KeyHelper : public boost::static_visitor<bool> {
        explicit KeyHelper(ItemVariant const *other_) : other(other_) {}

        template <typename T>
        bool operator()(SchemaItem<T> const &a) const {
            SchemaItem<T> const *b = boost::get<SchemaItem<T> >(other);
            return (b) && a.key == b->key;
        }

        ItemVariant const *other;
    };

    // Extracts field name from an ItemVariant
    struct NameHelper : public boost::static_visitor<std::string const &> {
        template <typename T>
        std::string const &operator()(SchemaItem<T> const &a) const {
            return a.field.getName();
        }
    };

    // Extracts field doc from an ItemVariant
    struct DocHelper : public boost::static_visitor<std::string const &> {
        template <typename T>
        std::string const &operator()(SchemaItem<T> const &a) const {
            return a.field.getDoc();
        }
    };

    // Extracts field units from an ItemVariant
    struct UnitsHelper : public boost::static_visitor<std::string const &> {
        template <typename T>
        std::string const &operator()(SchemaItem<T> const &a) const {
            return a.field.getUnits();
        }
    };

public:
    static bool compareKeys(ItemVariant const &a, ItemVariant const &b) {
        return boost::apply_visitor(KeyHelper(&b), a);
    }

    static bool compareNames(ItemVariant const &a, ItemVariant const &b) {
        return boost::apply_visitor(NameHelper(), a) == boost::apply_visitor(NameHelper(), b);
    }

    static bool compareDocs(ItemVariant const &a, ItemVariant const &b) {
        return boost::apply_visitor(DocHelper(), a) == boost::apply_visitor(DocHelper(), b);
    }

    static bool compareUnits(ItemVariant const &a, ItemVariant const &b) {
        return boost::apply_visitor(UnitsHelper(), a) == boost::apply_visitor(UnitsHelper(), b);
    }
};

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- SchemaImpl implementation ---------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

namespace detail {

//----- Finding a SchemaItem by field name ------------------------------------------------------------------

// This is easier to understand if you start reading from the bottom of this section, with
// SchemaImpl::find(std::string const &), then work your way up.

namespace {

// Given a SchemaItem for a regular field, look for a subfield with the given name.
// Return the index of the subfield (>= 0) on success, -1 on failure.
template <typename T>
inline int findNamedSubfield(
        SchemaItem<T> const &item, std::string const &name, char delimiter,
        boost::mpl::true_ *  // whether a match is possible based on the type of T; computed by caller
        ) {
    if (name.size() <= item.field.getName().size()) return -1;

    if (  // compare invocation is equivalent to "name.startswith(item.field.getName())" in Python
            name.compare(0, item.field.getName().size(), item.field.getName()) == 0 &&
            name[item.field.getName().size()] == delimiter) {
        int const position = item.field.getName().size() + 1;
        int const size = name.size() - position;
        int const nElements = item.field.getElementCount();
        for (int i = 0; i < nElements; ++i) {
            if (name.compare(position, size, Key<T>::subfields[i]) == 0) {
                return i;
            }
        }
    }
    return -1;
}

// This is an overload of findNamedSubfield that always fails; it's called when we
// know from the type of the field and subfield that they're incompatible.
template <typename T>
inline int findNamedSubfield(
        SchemaItem<T> const &item, std::string const &name, char delimiter,
        boost::mpl::false_ *  // whether a match is possible based on the type of T; computed by caller
        ) {
    return -1;
}

// Given a SchemaItem and a subfield index, make a new SchemaItem that corresponds to that
// subfield and put it in the result smart pointer.
template <typename T, typename U>
inline void makeSubfieldItem(
        SchemaItem<T> const &item, int index, char delimiter, std::unique_ptr<SchemaItem<U> > &result,
        boost::mpl::true_ *  // whether a match is possible based on the types of T and U; computed by caller
        ) {
    result.reset(new SchemaItem<U>(detail::Access::extractElement(item.key, index),
                                   Field<U>(join(item.field.getName(), Key<T>::subfields[index], delimiter),
                                            item.field.getDoc(), item.field.getUnits())));
}

// An overload of makeSubfieldItem that always fails because we know T and U aren't compatible.
template <typename T, typename U>
inline void makeSubfieldItem(
        SchemaItem<T> const &item, int index, char delimiter, std::unique_ptr<SchemaItem<U> > &result,
        boost::mpl::false_ *  // whether a match is possible based on the types of T and U; computed by caller
        ) {}

// This is a Variant visitation functor used to extract subfield items by name.
// For example, if we have a Point field "a", if we search the Schema for "a.x",
// we want to return a SchemaItem that makes it look like "a.x" is a full-fledged
// field in its own right.
template <typename U>
struct ExtractItemByName : public boost::static_visitor<> {
    explicit ExtractItemByName(std::string const &name_, char delimiter_)
            : delimiter(delimiter_), name(name_) {}

    template <typename T>
    void operator()(SchemaItem<T> const &item) const {
        // We want to find out if 'item' has a subfield whose fully-qualified name matches our
        // name data member.  But we also know that the subfield needs to have type U, and that
        // the field needs to have named subfields.
        // This typedef is boost::mpl::true_ if all the above is true, and boost::mpl::false_ otherwise.
        typedef typename boost::mpl::and_<std::is_same<U, typename Field<T>::Element>,
                                          boost::mpl::bool_<KeyBase<T>::HAS_NAMED_SUBFIELDS> >::type
                IsMatchPossible;
        // We use that type to dispatch one of the two overloads of findNamedSubfield.
        int n = findNamedSubfield(item, name, delimiter, (IsMatchPossible *)0);
        // If we have a match, we call another overloaded template to make the subfield.
        if (n >= 0) makeSubfieldItem(item, n, delimiter, result, (IsMatchPossible *)0);
    }

    char delimiter;
    std::string name;                                // name we're looking for
    mutable std::unique_ptr<SchemaItem<U> > result;  // where we put the result to signal that we're done
};

}  // namespace

// Here's the driver for the find-by-name algorithm.
template <typename T>
SchemaItem<T> SchemaImpl::find(std::string const &name) const {
    NameMap::const_iterator i = _names.lower_bound(name);
    if (i != _names.end()) {
        if (i->first == name) {
            // got an exact match; we're done if it has the right type, and dead if it doesn't.
            try {
                return boost::get<SchemaItem<T> const>(_items[i->second]);
            } catch (boost::bad_get &err) {
                throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                                  (boost::format("Field '%s' does not have the given type.") % name).str());
            }
        }
    }
    // We didn't get an exact match, but we might be searching for "a.x/a_x" and "a" might be a point field.
    // Because the names are sorted, we know we overshot it, so we work backwards.
    ExtractItemByName<T> extractor(name, getDelimiter());
    while (i != _names.begin()) {
        --i;
        boost::apply_visitor(extractor, _items[i->second]);  // see if the current item is a match
        if (extractor.result) return *extractor.result;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                      (boost::format("Field or subfield withname '%s' not found with type '%s'.") % name %
                       Field<T>::getTypeString())
                              .str());
}

//----- Finding a SchemaItem by key -------------------------------------------------------------------------

// This is easier to understand if you start reading from the bottom of this section, with
// SchemaImpl::find(Key<T> const &), then work your way up.

namespace {

// Given a SchemaItem for a regular field, look for a subfield with the given Key
// Return the index of the subfield (>= 0) on success, -1 on failure.
template <typename T, typename U>
inline int findKeySubfield(
        SchemaItem<T> const &item, Key<U> const &key,
        boost::mpl::true_ *  // whether a match is possible based on the types of T and U; computed by caller
        ) {
    int n = (key.getOffset() - item.key.getOffset()) / sizeof(U);
    if (n >= 0 && n < item.key.getElementCount()) {
        return n;
    }
    return -1;
}

// This is an overload of findKeySubfield that always fails; it's called when we
// know from the type of the field and subfield that they're incompatible.
template <typename T, typename U>
inline int findKeySubfield(
        SchemaItem<T> const &item, Key<U> const &key,
        boost::mpl::false_ *  // whether a match is possible based on the types of T and U; computed by caller
        ) {
    return -1;
}

// This is a Variant visitation functor used to extract subfield items by key.
template <typename U>
struct ExtractItemByKey : public boost::static_visitor<> {
    explicit ExtractItemByKey(Key<U> const &key_, char delimiter_) : delimiter(delimiter_), key(key_) {}

    template <typename T>
    void operator()(SchemaItem<T> const &item) const {
        // We want to find out if 'item' has a subfield whose  matches our key data member.
        // But we also know that the subfield needs to have type U.
        // This typedef is boost::mpl::true_ if the above is true, and boost::mpl::false_ otherwise.
        typedef typename boost::mpl::and_<std::is_same<U, typename Field<T>::Element>,
                                          boost::mpl::bool_<KeyBase<T>::HAS_NAMED_SUBFIELDS> >::type
                IsMatchPossible;
        // We use that type to dispatch one of the two overloads of findKeySubfield.
        int n = findKeySubfield(item, key, (IsMatchPossible *)0);
        // If we have a match, we call another overloaded template to make the subfield.
        // (this is the same  makeSubfieldItem used in ExtractItemByName, so it's defined up there)
        if (n >= 0) makeSubfieldItem(item, n, delimiter, result, (IsMatchPossible *)0);
    }

    char delimiter;
    Key<U> key;
    mutable std::unique_ptr<SchemaItem<U> > result;
};

}  // namespace

// Here's the driver for the find-by-key algorithm.  It's pretty similar to the find-by-name algorithm.
template <typename T>
SchemaItem<T> SchemaImpl::find(Key<T> const &key) const {
    OffsetMap::const_iterator i = _offsets.lower_bound(key.getOffset());
    if (i != _offsets.end()) {
        if (i->first == key.getOffset()) {
            try {
                return boost::get<SchemaItem<T> const>(_items[i->second]);
            } catch (boost::bad_get &err) {
                // just swallow the exception; this might be a subfield key that points to the beginning.
            }
        }
        // We didn't get an exact match, but we might be searching for a subfield.
        // Because the offsets are sorted, we know we overshot it, so we work backwards.
        ExtractItemByKey<T> extractor(key, getDelimiter());
        while (i != _offsets.begin()) {
            --i;
            boost::apply_visitor(extractor, _items[i->second]);
            if (extractor.result) return *extractor.result;
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
                return boost::get<SchemaItem<Flag> const>(_items[i->second]);
            } catch (boost::bad_get &err) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                                  (boost::format("Flag field with offset %d and bit %d not found.") %
                                   key.getOffset() % key.getBit())
                                          .str());
            }
        }
    }
    // Flag keys are never subfields, so we require an exact match.
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
inline int findKey(SchemaImpl::OffsetMap const &offsets, SchemaImpl::FlagMap const &flags, Key<T> const &key,
                   bool throwIfMissing = true) {
    SchemaImpl::OffsetMap::const_iterator i = offsets.find(key.getOffset());
    if (i == offsets.end()) {
        if (throwIfMissing) {
            throw LSST_EXCEPT(pex::exceptions::NotFoundError,
                              (boost::format("Key of type %s with offset %d not found in Schema") %
                               Field<T>::getTypeString() % key.getOffset())
                                      .str());
        } else {
            return -1;
        }
    }
    return i->second;
}

// Like the above, but special-cased for Flag
inline int findKey(SchemaImpl::OffsetMap const &offsets, SchemaImpl::FlagMap const &flags,
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
            return -1;
        }
    }
    return i->second;
}

}  // namespace

template <typename T>
void SchemaImpl::replaceField(Key<T> const &key, Field<T> const &field) {
    NameMap::iterator j = _names.find(field.getName());
    SchemaItem<T> *item = 0;
    if (j != _names.end()) {
        // The field name is already present in the Schema; see if it's the one we're replacing.
        // If we can get the old item with this, we don't need to update the name map at all.
        item = boost::get<SchemaItem<T> >(&_items[j->second]);
        if (!item || key != item->key) {
            throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterError,
                    (boost::format("Field with name '%s' already present in schema with a different key.") %
                     field.getName())
                            .str());
        }
    }
    if (!item) {  // Need to find the original item by key, since it's a new name.
        int index = findKey(_offsets, _flags, key);
        item = boost::get<SchemaItem<T> >(&_items[index]);
        if (!item) {
            throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                              (boost::format("Incorrect key type '%s'.") % key).str());
        }
        j = _names.find(item->field.getName());
        _names.insert(j, std::pair<std::string, int>(field.getName(), j->second));
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
    SchemaItem<T> const *cmpItem = 0;
    int index = findKey(_offsets, _flags, item.key, false);
    if (index >= 0) {
        cmpItem = boost::get<SchemaItem<T> >(&_items[index]);
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
        for (NameMap::const_iterator i = _names.begin(); i != _names.end(); ++i) {
            std::size_t sep = i->first.find(getDelimiter());
            if (sep == std::string::npos) {
                result.insert(result.end(), i->first);
            } else {
                result.insert(result.end(), i->first.substr(0, sep));
            }
        }
    } else {
        for (NameMap::const_iterator i = _names.begin(); i != _names.end(); ++i) {
            result.insert(result.end(), i->first);
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
    return addFieldImpl(sizeof(typename Field<std::string>::Element), field.getElementCount(), field, doReplace);
}

template <typename T>
Key<T> SchemaImpl::addField(Field<T> const &field, bool doReplace) {
    return addFieldImpl(sizeof(typename Field<T>::Element), field.getElementCount(), field, doReplace);
}

Key<Flag> SchemaImpl::addField(Field<Flag> const &field, bool doReplace) {
    static int const ELEMENT_SIZE = sizeof(Field<Flag>::Element);
    std::pair<NameMap::iterator, bool> result =
            _names.insert(std::pair<std::string, int>(field.getName(), _items.size()));
    if (!result.second) {
        if (doReplace) {
            SchemaItem<Flag> *item = boost::get<SchemaItem<Flag> >(&_items[result.first->second]);
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
        if (_lastFlagField < 0 || _lastFlagBit >= ELEMENT_SIZE * 8) {
            int padding = ELEMENT_SIZE - _recordSize % ELEMENT_SIZE;
            if (padding != ELEMENT_SIZE) {
                _recordSize += padding;
            }
            _lastFlagField = _recordSize;
            _lastFlagBit = 0;
            _recordSize += field.getElementCount() * ELEMENT_SIZE;
        }
        SchemaItem<Flag> item(detail::Access::makeKey(_lastFlagField, _lastFlagBit), field);
        ++_lastFlagBit;
        _flags.insert(std::pair<std::pair<int, int>, int>(
                std::make_pair(item.key.getOffset(), item.key.getBit()), _items.size()));
        _items.emplace_back(item);
        return item.key;
    }
}

template <typename T>
Key<T> SchemaImpl::addFieldImpl(int elementSize, int elementCount, Field<T> const &field, bool doReplace) {
    std::pair<NameMap::iterator, bool> result =
            _names.insert(std::pair<std::string, int>(field.getName(), _items.size()));
    if (!result.second) {
        if (doReplace) {
            SchemaItem<T> *item = boost::get<SchemaItem<T> >(&_items[result.first->second]);
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
        int padding = elementSize - _recordSize % elementSize;
        if (padding != elementSize) {
            _recordSize += padding;
        }
        SchemaItem<T> item(detail::Access::makeKey(field, _recordSize), field);
        _recordSize += elementCount * elementSize;
        _offsets.insert(std::pair<int, int>(item.key.getOffset(), _items.size()));
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

Schema::Schema(Schema const &other) : _impl(other._impl), _aliases(other._aliases) {}
// Delegate to copy constructor  for backwards compatibility
Schema::Schema(Schema &&other) : Schema(other) {}

Schema &Schema::operator=(Schema const&) = default;
Schema &Schema::operator=(Schema&&) = default;
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
    typedef void result_type;

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
    for (auto iter = schema.getAliasMap()->begin(); iter != schema.getAliasMap()->end(); ++iter) {
        os << "    '" << iter->first << "'->'" << iter->second << "'\n";
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
