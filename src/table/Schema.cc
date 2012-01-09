#include <list>
#include <stdexcept>

#include "boost/make_shared.hpp"
#include "boost/utility/enable_if.hpp"
#include "boost/type_traits/is_same.hpp"
#include "boost/mpl/and.hpp"
#include "boost/mpl/bool.hpp"
#include "boost/scoped_ptr.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw { namespace table {

//----- Miscellaneous Utilities -----------------------------------------------------------------------------

namespace {

std::string join(std::string const & a, std::string const & b) {
    std::string full;
    full.reserve(a.size() + b.size() + 1);
    full += a;
    full.push_back('.');
    full += b;
    return full;
}

struct Stream {
    
    typedef void result_type;
   
    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        *os << "    " << item.field << ",\n";
    }

    explicit Stream(std::ostream * os_) : os(os_) {}

    std::ostream * os;
};

struct ExtractOffset : public boost::static_visitor<int> {

    typedef int result_type;

    template <typename T>
    result_type operator()(SchemaItem<T> const & item) const {
        return item.key.getOffset();
    }

    result_type operator()(boost::blank const &) const { return 0; }

    result_type operator()(detail::SchemaImpl::ItemVariant const & v) const {
        return boost::apply_visitor(*this, v);
    }

};

struct CompareItemKeys {

    typedef bool result_type;

    struct Helper : public boost::static_visitor<bool> {
        
        result_type operator()(boost::blank const &) const { return false; }

        template <typename T>
        bool operator()(SchemaItem<T> const & a) const {
            SchemaItem<T> const * b = boost::get< SchemaItem<T> >(other);
            return (b) && a.key == b->key;
        }

        explicit Helper(detail::SchemaImpl::ItemVariant const * other_) : other(other_) {}

        detail::SchemaImpl::ItemVariant const * other;
    };

    result_type operator()(
        detail::SchemaImpl::ItemVariant const & a,
        detail::SchemaImpl::ItemVariant const & b
    ) const {
        return boost::apply_visitor(Helper(&b), a);
    }

};

/*
 * This template class provides the implementation for the ExtractItemByX functors,
 * which construct SchemaItems on the fly for subfields.  The default implementation
 * does nothing and always fails.
 */
template <typename T, typename U, typename V=typename Field<T>::Element,
          bool HasSubfields = KeyBase<T>::HAS_NAMED_SUBFIELDS>
struct ExtractItem {

    static void finish(int index, boost::scoped_ptr< SchemaItem<U> > & result, SchemaItem<T> const & item) {}

    static int apply(std::string const & name, SchemaItem<T> const & item) { return -1; }

    static int apply(int offset, SchemaItem<T> const & item) { return -1; }

};

/*
 * This specialization of ExtractItem, which is the only one that can succeed, only matches
 * when U is the Element type of T and T has named subfields.
 */
template <typename T, typename U>
struct ExtractItem<T,U,U,true> {

    static void finish(int index, boost::scoped_ptr< SchemaItem<U> > & result, SchemaItem<T> const & item) {
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

    static int apply(
        std::string const & name, SchemaItem<T> const & item
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

    static int apply(
        int offset, SchemaItem<T> const & item
    ) {
        int n = (offset - item.key.getOffset()) / sizeof(U);
        if (n >= 0 && n < item.key.getElementCount()) {
            return n;
        }
        return -1;
    }

};

template <typename U>
struct ExtractItemByName : public boost::static_visitor<> {
    
    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        int n = ExtractItem<T,U>::apply(name, item);
        if (n >= 0) ExtractItem<T,U>::finish(n, result, item);
    }

    void operator()(boost::blank const &) const {}

    explicit ExtractItemByName(std::string const & name_) : name(name_) {}

    std::string name;
    mutable boost::scoped_ptr< SchemaItem<U> > result;
};

template <typename U>
struct ExtractItemByOffset : public boost::static_visitor<> {
    
    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        int n = ExtractItem<T,U>::apply(offset, item);
        if (n >= 0) ExtractItem<T,U>::finish(n, result, item);
    }

    void operator()(boost::blank const &) const {}

    explicit ExtractItemByOffset(int offset_) : offset(offset_) {}

    int offset;
    mutable boost::scoped_ptr< SchemaItem<U> > result;
};

} // anonymous

//----- SchemaImpl implementation ---------------------------------------------------------------------------

namespace detail {

template <typename T>
SchemaItem<T> SchemaImpl::find(std::string const & name) const {
    NameMap::const_iterator i = _names.lower_bound(name);
    if (i != _names.end()) {
        if (i->first == name) {
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
    ExtractItemByName<T> extractor(name);
    if (i != _names.begin()) {
        --i;
        boost::apply_visitor(extractor, _items[i->second]);
        if (extractor.result) return *extractor.result;
    }
    throw LSST_EXCEPT(
        lsst::pex::exceptions::NotFoundException,
        (boost::format("Field or subfield with name '%s' not found with the given type.") % name).str()
    );
}

template <typename T>
SchemaItem<T> SchemaImpl::find(Key<T> const & key) const {
    typedef boost::transform_iterator<ExtractOffset,ItemContainer::const_iterator> Iterator;
    int const offset = key.getOffset();
    Iterator i = std::lower_bound(
        Iterator(_items.begin()),
        Iterator(_items.end()), 
        offset
    );
    if (*i == offset) {
        try {
            return boost::get< SchemaItem<T> const >(*i.base());
        } catch (boost::bad_get & err) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                "Field with the given key does not have the given type."
            );
        }
    }
    ExtractItemByOffset<T> extractor(offset);
    if (i.base() != _items.begin()) {
        --i;
        boost::apply_visitor(extractor, *i.base());
        if (extractor.result) return *extractor.result;
    }
    throw LSST_EXCEPT(
        lsst::pex::exceptions::NotFoundException,
        "Field or subfield with the given key not found with the given type."
    );

    // TODO
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
Key<T> SchemaImpl::addField(Field<T> const & field) {
    static int const ELEMENT_SIZE = sizeof(typename Field<T>::Element);
    if (!_names.insert(std::pair<std::string,int>(field.getName(), _items.size())).second) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Field with name '%s' already present in schema.") % field.getName()).str()
        );
    }
    int padding = ELEMENT_SIZE - _recordSize % ELEMENT_SIZE;
    if (padding != ELEMENT_SIZE) {
        _recordSize += padding;
    }
    SchemaItem<T> item(detail::Access::makeKey(field, _recordSize), field);
    _recordSize += field.getElementCount() * ELEMENT_SIZE;
    _items.push_back(item);
    return item.key;
}

Key<Flag> SchemaImpl::addField(Field<Flag> const & field) {
    static int const ELEMENT_SIZE = sizeof(Field<Flag>::Element);
    if (!_names.insert(std::pair<std::string,int>(field.getName(), _items.size())).second) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Field with name '%s' already present in schema.") % field.getName()).str()
        );
    }
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
    _items.push_back(item);
    return item.key;
}

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
        typedef boost::transform_iterator<ExtractOffset,ItemContainer::iterator> Iterator;
        int const offset = key.getOffset();
        Iterator i = std::lower_bound(
            Iterator(_items.begin()),
            Iterator(_items.end()),
            offset
        );
        if (*i != offset) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundException,
                "Key not found in Schema."
            );
        }
        item = boost::get< SchemaItem<T> >(&(*i.base()));
        if (!item) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                "Field with the given key does not have the given type."
            );
        }
        j = _names.find(item->field.getName());
        _names.insert(j, std::pair<std::string,int>(field.getName(), j->second));
        _names.erase(j);
    }
    item->field = field;
}

} // namespace detail

//----- Schema implementation -------------------------------------------------------------------------------

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
Key<T> Schema::addField(Field<T> const & field) {
    _edit();
    return _impl->addField(field);
}

template <typename T>
void Schema::replaceField(Key<T> const & key, Field<T> const & field) {
    _edit();
    _impl->replaceField(key, field);
}

bool Schema::operator==(Schema const & other) const {
    if (_impl == other._impl) return true;
    if (_impl->getItems().size() != other._impl->getItems().size()) return false;
    return std::equal(
        _impl->getItems().begin(), _impl->getItems().end(),
        other._impl->getItems().begin(),
        CompareItemKeys()
    );
}

Schema::Schema(bool hasTree) : _impl(boost::make_shared<Impl>(hasTree)) {}

std::ostream & operator<<(std::ostream & os, Schema const & schema) {
    os << "Schema(\n";
    schema.forEach(Stream(&os));
    return os << ")\n";
}

//----- SubSchema implementation ----------------------------------------------------------------------------

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

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUT(r, data, elem)                               \
    template Key< elem > Schema::addField(Field< elem > const &);            \
    template SchemaItem< elem > Schema::find(std::string const & ) const; \
    template SchemaItem< elem > Schema::find(Key< elem > const & ) const; \
    template void Schema::replaceField(Key< elem > const &, Field< elem > const &); \
    template SchemaItem< elem > SubSchema::find(std::string const & ) const; \

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUT, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
