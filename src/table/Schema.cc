#include <list>
#include <stdexcept>

#include "boost/make_shared.hpp"
#include "boost/type_traits/is_same.hpp"
#include "boost/mpl/and.hpp"
#include "boost/mpl/bool.hpp"
#include "boost/scoped_ptr.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/iterator/filter_iterator.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/io/FitsReader.h"

namespace lsst { namespace afw { namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- Miscellaneous Utilities -----------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

namespace {

// Concatenate two strings with a period between them.
std::string join(std::string const & a, std::string const & b) {
    std::string full;
    full.reserve(a.size() + b.size() + 1);
    full += a;
    full.push_back('.');
    full += b;
    return full;
}

// Functor to compare two ItemVariants for Key equality.
class ItemFunctors {

    typedef detail::SchemaImpl::ItemVariant ItemVariant;

    // Compares keys - must be initialized with one ItemVariant and passed the other.
    struct KeyHelper : public boost::static_visitor<bool> {

        template <typename T>
        bool operator()(SchemaItem<T> const & a) const {
            SchemaItem<T> const * b = boost::get< SchemaItem<T> >(other);
            return (b) && a.key == b->key;
        }

        explicit KeyHelper(ItemVariant const * other_) : other(other_) {}

        ItemVariant const * other;
    };

    // Extracts field name from an ItemVariant
    struct NameHelper : public boost::static_visitor<std::string const &> {
        template <typename T>
        std::string const & operator()(SchemaItem<T> const & a) const { return a.field.getName(); }
    };

    // Extracts field doc from an ItemVariant
    struct DocHelper : public boost::static_visitor<std::string const &> {
        template <typename T>
        std::string const & operator()(SchemaItem<T> const & a) const { return a.field.getDoc(); }
    };

    // Extracts field units from an ItemVariant
    struct UnitsHelper : public boost::static_visitor<std::string const &> {
        template <typename T>
        std::string const & operator()(SchemaItem<T> const & a) const { return a.field.getUnits(); }
    };

public:

    static bool compareKeys(ItemVariant const & a, ItemVariant const & b) {
        return boost::apply_visitor(KeyHelper(&b), a);
    }

    static bool compareNames(ItemVariant const & a, ItemVariant const & b) {
        return boost::apply_visitor(NameHelper(), a) == boost::apply_visitor(NameHelper(), b);
    }

    static bool compareDocs(ItemVariant const & a, ItemVariant const & b) {
        return boost::apply_visitor(DocHelper(), a) == boost::apply_visitor(DocHelper(), b);
    }

    static bool compareUnits(ItemVariant const & a, ItemVariant const & b) {
        return boost::apply_visitor(UnitsHelper(), a) == boost::apply_visitor(UnitsHelper(), b);
    }

};

} // anonymous

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
    SchemaItem<T> const & item,
    std::string const & name,
    boost::mpl::true_ * // whether a match is possible based on the type of T; computed by caller
) {
    if (name.size() <= item.field.getName().size()) return -1;

    if ( // compare invocation is equivalent to "name.startswith(item.field.getName())" in Python
        name.compare(0, item.field.getName().size(), item.field.getName()) == 0
        && name[item.field.getName().size()] == '.'
    ) {
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
    SchemaItem<T> const & item,
    std::string const & name,
    boost::mpl::false_ * // whether a match is possible based on the type of T; computed by caller
) {
    return -1;
}

// Given a SchemaItem and a subfield index, make a new SchemaItem that corresponds to that
// subfield and put it in the result smart pointer.
template <typename T, typename U>
inline void makeSubfieldItem(
    SchemaItem<T> const & item, int index,
    boost::scoped_ptr< SchemaItem<U> > & result,
    boost::mpl::true_ * // whether a match is possible based on the types of T and U; computed by caller
) {
    result.reset(
        new SchemaItem<U>(
            detail::Access::extractElement(item.key, index),
            Field<U>(
                join(item.field.getName(), Key<T>::subfields[index]),
                item.field.getDoc(),
                item.field.getUnits()
            )
        )
    );
}

// An overload of makeSubfieldItem that always fails because we know T and U aren't compatible.
template <typename T, typename U>
inline void makeSubfieldItem(
    SchemaItem<T> const & item, int index,
    boost::scoped_ptr< SchemaItem<U> > & result,
    boost::mpl::false_ * // whether a match is possible based on the types of T and U; computed by caller
) {}

// This is a Variant visitation functor used to extract subfield items by name.
// For example, if we have a Point field "a", if we search the Schema for "a.x",
// we want to return a SchemaItem that makes it look like "a.x" is a full-fledged
// field in its own right.
template <typename U>
struct ExtractItemByName : public boost::static_visitor<> {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        // We want to find out if 'item' has a subfield whose fully-qualified name matches our
        // name data member.  But we also know that the subfield needs to have type U, and that
        // the field needs to have named subfields.
        // This typedef is boost::mpl::true_ if all the above is true, and boost::mpl::false_ otherwise.
        typedef typename boost::mpl::and_<
            boost::is_same<U, typename Field<T>::Element>,
            boost::mpl::bool_<KeyBase<T>::HAS_NAMED_SUBFIELDS>
        >::type IsMatchPossible;
        // We use that type to dispatch one of the two overloads of findNamedSubfield.
        int n = findNamedSubfield(item, name, (IsMatchPossible*)0);
        // If we have a match, we call another overloaded template to make the subfield.
        if (n >= 0) makeSubfieldItem(item, n, result, (IsMatchPossible*)0);
    }

    explicit ExtractItemByName(std::string const & name_) : name(name_) {}

    std::string name; // name we're looking for
    mutable boost::scoped_ptr< SchemaItem<U> > result; // where we put the result to signal that we're done
};

} // anonymous

// Here's the driver for the find-by-name algorithm.
template <typename T>
SchemaItem<T> SchemaImpl::find(std::string const & name) const {
    NameMap::const_iterator i = _names.lower_bound(name);
    if (i != _names.end()) {
        if (i->first == name) {
            // got an exact match; we're done if it has the right type, and dead if it doesn't.
            try {
                return boost::get< SchemaItem<T> const >(_items[i->second]);
            } catch (boost::bad_get & err) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterException,
                    (boost::format("Field '%s' does not have the given type.") % name).str()
                );
            }
        }
    }
    // We didn't get an exact match, but we might be searching for "a.x" and "a" might be a point field.
    // Because the names are sorted, we know we overshot it, so we work backwards.
    ExtractItemByName<T> extractor(name);
    while (i != _names.begin()) {
        --i;
        boost::apply_visitor(extractor, _items[i->second]); // see if the current item is a match
        if (extractor.result) return *extractor.result;
    }
    throw LSST_EXCEPT(
        lsst::pex::exceptions::NotFoundException,
        (boost::format("Field or subfield withname '%s' not found with type '%s'.")
         % name % Field<T>::getTypeString()).str()
    );
}

//----- Finding a SchemaItem by key -------------------------------------------------------------------------

// This is easier to understand if you start reading from the bottom of this section, with
// SchemaImpl::find(Key<T> const &), then work your way up.

namespace {

// Given a SchemaItem for a regular field, look for a subfield with the given Key
// Return the index of the subfield (>= 0) on success, -1 on failure.
template <typename T, typename U>
inline int findKeySubfield(
    SchemaItem<T> const & item,
    Key<U> const & key,
    boost::mpl::true_ * // whether a match is possible based on the types of T and U; computed by caller
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
    SchemaItem<T> const & item,
    Key<U> const & key,
    boost::mpl::false_ * // whether a match is possible based on the types of T and U; computed by caller
) {
    return -1;
}

// This is a Variant visitation functor used to extract subfield items by key.
template <typename U>
struct ExtractItemByKey : public boost::static_visitor<> {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        // We want to find out if 'item' has a subfield whose  matches our key data member.
        // But we also know that the subfield needs to have type U.
        // This typedef is boost::mpl::true_ if the above is true, and boost::mpl::false_ otherwise.
        typedef typename boost::mpl::and_<
            boost::is_same<U, typename Field<T>::Element>,
            boost::mpl::bool_<KeyBase<T>::HAS_NAMED_SUBFIELDS>
        >::type IsMatchPossible;
        // We use that type to dispatch one of the two overloads of findKeySubfield.
        int n = findKeySubfield(item, key, (IsMatchPossible*)0);
        // If we have a match, we call another overloaded template to make the subfield.
        // (this is the same  makeSubfieldItem used in ExtractItemByName, so it's defined up there)
        if (n >= 0) makeSubfieldItem(item, n, result, (IsMatchPossible*)0);
    }

    explicit ExtractItemByKey(Key<U> const & key_) : key(key_) {}

    Key<U> key;
    mutable boost::scoped_ptr< SchemaItem<U> > result;
};

} // anonymous.

// Here's the driver for the find-by-key algorithm.  It's pretty similar to the find-by-name algorithm.
template <typename T>
SchemaItem<T> SchemaImpl::find(Key<T> const & key) const {
    OffsetMap::const_iterator i = _offsets.lower_bound(key.getOffset());
    if (i->first == key.getOffset()) {
        try {
            return boost::get< SchemaItem<T> const >(_items[i->second]);
        } catch (boost::bad_get & err) {
            // just swallow the exception; this might be a subfield key that points to the beginning.
        }
    }
    // We didn't get an exact match, but we might be searching for a subfield.
    // Because the offsets are sorted, we know we overshot it, so we work backwards.
    ExtractItemByKey<T> extractor(key);
    while (i != _offsets.begin()) {
        --i;
        boost::apply_visitor(extractor, _items[i->second]);
        if (extractor.result) return *extractor.result;
    }
    throw LSST_EXCEPT(
        lsst::pex::exceptions::NotFoundException,
        (boost::format("Field or subfield with offset %d not found with type '%s'.")
         % key.getOffset() % Field<T>::getTypeString()).str()
    );
}

// We handle Flag fields separately when searching for keys, because their keys aren't like the others.
SchemaItem<Flag> SchemaImpl::find(Key<Flag> const & key) const {
    FlagMap::const_iterator i = _flags.lower_bound(std::make_pair(key.getOffset(), key.getBit()));
    if (i->first.first == key.getOffset() && i->first.second == key.getBit()) {
        try {
            return boost::get< SchemaItem<Flag> const >(_items[i->second]);
        } catch (boost::bad_get & err) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundException,
                (boost::format("Flag field with offset %d and bit %d not found.")
                 % key.getOffset() % key.getBit()).str()
            );
        }
    }
    // Flag keys are never subfields, so we require an exact match.
    throw LSST_EXCEPT(
        lsst::pex::exceptions::NotFoundException,
        (boost::format("Flag field with offset %d and bit %d not found.")
         % key.getOffset() % key.getBit()).str()
    );
}

//----- Replacing an existing SchemaItem --------------------------------------------------------------------

// This is easier to understand if you start reading from the bottom of this section, with
// SchemaImpl::replaceField, then work your way up.

namespace {

// Find an exact SchemaItem by key ('exact' means no subfields, unlike the find member function above)
// Return the index into the item container.
template <typename T>
inline int findKey(
    SchemaImpl::OffsetMap const & offsets,
    SchemaImpl::FlagMap const & flags,
    Key<T> const & key,
    bool throwIfMissing = true
) {
    SchemaImpl::OffsetMap::const_iterator i = offsets.find(key.getOffset());
    if (i == offsets.end()) {
        if (throwIfMissing) {
            throw LSST_EXCEPT(
                pex::exceptions::NotFoundException,
                (boost::format("Key of type %s with offset %d not found in Schema")
                 % Field<T>::getTypeString() % key.getOffset()).str()
            );
        } else {
            return -1;
        }
    }
    return i->second;
}

// Like the above, but special-cased for Flag
inline int findKey(
    SchemaImpl::OffsetMap const & offsets,
    SchemaImpl::FlagMap const & flags,
    Key<Flag> const & key,
    bool throwIfMissing = true
) {
    SchemaImpl::FlagMap::const_iterator i = flags.find(std::make_pair(key.getOffset(), key.getBit()));
    if (i == flags.end()) {
        if (throwIfMissing) {
            throw LSST_EXCEPT(
                pex::exceptions::NotFoundException,
                (boost::format("Key of type Flag with offset %d and bit %d not found in Schema")
                 % key.getOffset() % key.getBit()).str()
            );
        } else {
            return -1;
        }
    }
    return i->second;
}

} // anonymous

template <typename T>
void SchemaImpl::replaceField(Key<T> const & key, Field<T> const & field) {
    NameMap::iterator j = _names.find(field.getName());
    SchemaItem<T> * item = 0;
    if (j != _names.end()) {
        // The field name is already present in the Schema; see if it's the one we're replacing.
        // If we can get the old item with this, we don't need to update the name map at all.
        item = boost::get< SchemaItem<T> >(&_items[j->second]);
        if (!item || key != item->key) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Field with name '%s' already present in schema with a different key.")
                 % field.getName()).str()
            );
        }
    }
    if (!item) { // Need to find the original item by key, since it's a new name.
        int index = findKey(_offsets, _flags, key);
        item = boost::get< SchemaItem<T> >(&_items[index]);
        if (!item) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Incorrect key type '%s'.") % key).str()
            );
        }
        j = _names.find(item->field.getName());
        _names.insert(j, std::pair<std::string,int>(field.getName(), j->second));
        _names.erase(j);
    }
    item->field = field;
}

//----- Other SchemaImpl things -----------------------------------------------------------------------------

template <typename T>
int SchemaImpl::contains(SchemaItem<T> const & item, int flags) const {
    if (!(flags & Schema::EQUAL_KEYS)) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicErrorException,
            "Can only check whether item is in schema if flags & EQUAL_KEYS"
        );
    }
    SchemaItem<T> const * cmpItem = 0;
    int index = findKey(_offsets, _flags, item.key, false);
    if (index >= 0) {
        cmpItem = boost::get< SchemaItem<T> >(&_items[index]);
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
            std::size_t dot = i->first.find('.');
            if (dot == std::string::npos) {
                result.insert(result.end(), i->first);
            } else {
                result.insert(result.end(), i->first.substr(0, dot));
            }
        }
    } else {
        for (NameMap::const_iterator i = _names.begin(); i != _names.end(); ++i) {
            result.insert(result.end(), i->first);
        }
    }
    return result;
}

std::set<std::string> SchemaImpl::getNames(bool topOnly, std::string const & prefix) const {
    std::set<std::string> result;
    if (topOnly) {
        for (NameMap::const_iterator i = _names.lower_bound(prefix); i != _names.end(); ++i) {
            if (i->first.compare(0, prefix.size(), prefix) != 0) break;
            std::size_t dot = i->first.find('.', prefix.size() + 1);
            if (dot == std::string::npos) {
                result.insert(
                    result.end(),
                    i->first.substr(prefix.size() + 1, i->first.size() - prefix.size())
                );
            } else {
                result.insert(
                    result.end(),
                    i->first.substr(prefix.size() + 1, dot - prefix.size() - 1)
                );
            }
        }
    } else {
        for (NameMap::const_iterator i = _names.lower_bound(prefix); i != _names.end(); ++i) {
            if (i->first.compare(0, prefix.size(), prefix) != 0) break;
            result.insert(
                result.end(),
                i->first.substr(prefix.size() + 1, i->first.size() - prefix.size() - 1)
            );
        }
    }
    return result;
}

template <typename T>
Key<T> SchemaImpl::addField(Field<T> const & field, bool doReplace) {
    static int const ELEMENT_SIZE = sizeof(typename Field<T>::Element);
    std::pair<NameMap::iterator,bool> result
        = _names.insert(std::pair<std::string,int>(field.getName(), _items.size()));
    if (!result.second) {
        if (doReplace) {
            SchemaItem<T> * item = boost::get< SchemaItem<T> >(&_items[result.first->second]);
            if (!item) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterException,
                    (boost::format("Cannot replace field with name '%s' because types differ.")
                     % field.getName()).str()
                );
            }
            if (item->field.getElementCount() != field.getElementCount()) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterException,
                    (boost::format("Cannot replace field with name '%s' because sizes differ.")
                     % field.getName()).str()
                );
            }
            item->field = field;
            return item->key;
        } else {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Field with name '%s' already present in schema.") % field.getName()).str()
            );
        }
    } else {
        int padding = ELEMENT_SIZE - _recordSize % ELEMENT_SIZE;
        if (padding != ELEMENT_SIZE) {
            _recordSize += padding;
        }
        SchemaItem<T> item(detail::Access::makeKey(field, _recordSize), field);
        _recordSize += field.getElementCount() * ELEMENT_SIZE;
        _offsets.insert(std::pair<int,int>(item.key.getOffset(), _items.size()));
        _items.push_back(item);
        return item.key;
    }
}

Key<Flag> SchemaImpl::addField(Field<Flag> const & field, bool doReplace) {
    static int const ELEMENT_SIZE = sizeof(Field<Flag>::Element);
    std::pair<NameMap::iterator,bool> result
        = _names.insert(std::pair<std::string,int>(field.getName(), _items.size()));
    if (!result.second) {
        if (doReplace) {
            SchemaItem<Flag> * item = boost::get< SchemaItem<Flag> >(&_items[result.first->second]);
            if (!item) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterException,
                    (boost::format("Cannot replace field with name '%s' because types differ.")
                     % field.getName()).str()
                );
            }
            item->field = field;
            return item->key;
        } else {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Field with name '%s' already present in schema.") % field.getName()).str()
            );
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
        _flags.insert(
            std::pair<std::pair<int,int>,int>(
                std::make_pair(item.key.getOffset(), item.key.getBit()),
                _items.size()
            )
        );
        _items.push_back(item);
        return item.key;
    }
}

} // namespace detail

//-----------------------------------------------------------------------------------------------------------
//----- Schema implementation -------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

Schema::Schema() : _impl(boost::make_shared<Impl>()) {};

Schema::Schema(daf::base::PropertyList & metadata, bool stripMetadata) : _impl(boost::make_shared<Impl>()) {
    io::FitsReader::_readSchema(*this, metadata, stripMetadata);
}

Schema::Schema(daf::base::PropertyList const & metadata) : _impl(boost::make_shared<Impl>()) {
    io::FitsReader::_readSchema(*this, const_cast<daf::base::PropertyList &>(metadata), false);
}

void Schema::_edit() {
    if (!_impl.unique()) {
        boost::shared_ptr<Impl> data(boost::make_shared<Impl>(*_impl));
        _impl.swap(data);
    }
}

std::set<std::string> Schema::getNames(bool topOnly) const {
    return _impl->getNames(topOnly);
}

template <typename T>
SchemaItem<T> Schema::find(std::string const & name) const {
    return _impl->find<T>(name);
}

template <typename T>
SchemaItem<T> Schema::find(Key<T> const & key) const {
    return _impl->find(key);
}

template <typename T>
Key<T> Schema::addField(Field<T> const & field, bool doReplace) {
    _edit();
    return _impl->addField(field, doReplace);
}

template <typename T>
void Schema::replaceField(Key<T> const & key, Field<T> const & field) {
    _edit();
    _impl->replaceField(key, field);
}

int Schema::contains(Schema const & other, int flags) const {
    if (_impl == other._impl) return flags;
    if (_impl->getItems().size() < other._impl->getItems().size()) return 0;
    int result = flags;
    for (
        Impl::ItemContainer::const_iterator i1 = _impl->getItems().begin(),
            i2 = other._impl->getItems().begin();
        i2 != other._impl->getItems().end();
        ++i1, ++i2
    ) {
        if ((result & EQUAL_KEYS) && !ItemFunctors::compareKeys(*i1, *i2)) result &= ~EQUAL_KEYS;
        if ((result & EQUAL_NAMES) && !ItemFunctors::compareNames(*i1, *i2)) result &= ~EQUAL_NAMES;
        if ((result & EQUAL_DOCS) && !ItemFunctors::compareDocs(*i1, *i2)) result &= ~EQUAL_DOCS;
        if ((result & EQUAL_UNITS) && !ItemFunctors::compareUnits(*i1, *i2)) result &= ~EQUAL_UNITS;
        if (!result) break;
    }

    return result;
}

int Schema::compare(Schema const & other, int flags) const {
    return _impl->getItems().size() == other._impl->getItems().size() ? contains(other, flags) : 0;
}

template <typename T>
int Schema::contains(SchemaItem<T> const & item, int flags) const {
    return _impl->contains(item, flags);
}

//----- Stringification -------------------------------------------------------------------------------------

namespace {

// Schema::forEach functor used for stringificationx
struct Stream {

    typedef void result_type;

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        *os << "    (" << item.field << ", " << item.key << "),\n";
    }

    explicit Stream(std::ostream * os_) : os(os_) {}

    std::ostream * os;
};

} // anonymous

std::ostream & operator<<(std::ostream & os, Schema const & schema) {
    os << "Schema(\n";
    schema.forEach(Stream(&os));
    return os << ")\n";
}

//-----------------------------------------------------------------------------------------------------------
//----- SubSchema implementation ----------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

template <typename T>
SchemaItem<T> SubSchema::find(std::string const & name) const {
    return _impl->find<T>(join(_name, name));
}

SubSchema SubSchema::operator[](std::string const & name) const {
    return SubSchema(_impl, join(_name, name));
}

std::set<std::string> SubSchema::getNames(bool topOnly) const {
    return _impl->getNames(topOnly, _name);
}

//-----------------------------------------------------------------------------------------------------------
//----- Explicit instantiation ------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// Note: by instantiating the public functions below, we also instantiate a lot of the private
// implementation functions.  If you move some of those to a different source file, you'll need
// more explicit instantiation.

#define INSTANTIATE_LAYOUT(r, data, elem)                               \
    template Key< elem > Schema::addField(Field< elem > const &, bool);            \
    template SchemaItem< elem > Schema::find(std::string const & ) const; \
    template SchemaItem< elem > Schema::find(Key< elem > const & ) const; \
    template int Schema::contains(SchemaItem< elem > const &, int) const;   \
    template void Schema::replaceField(Key< elem > const &, Field< elem > const &); \
    template SchemaItem< elem > SubSchema::find(std::string const & ) const; \

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUT, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
