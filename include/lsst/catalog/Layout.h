// -*- c++ -*-
#ifndef CATALOG_Layout_h_INCLUDED
#define CATALOG_Layout_h_INCLUDED

#include <vector>
#include <list>
#include <set>
#include <string>
#include <stdexcept>

#include "boost/shared_ptr.hpp"
#include "boost/compressed_pair.hpp"
#include "boost/type_traits/is_same.hpp"

#include "lsst/ndarray.h"
#include "lsst/catalog/Field.h"

namespace lsst { namespace catalog {

namespace detail {
class LayoutImpl;
} // namespace detail

class Layout;
class LayoutBuilder;
class Table;
class ColumnView;

template <typename T>
class Key {
public:

    template <typename U>
    bool operator==(Key<U> const & other) const {
        return boost::is_same<T,U>::value && _data.first() == other._data.first();
    }

    template <typename U>
    bool operator!=(Key<U> const & other) const {
        return boost::is_same<T,U>::value && _data.first() != other._data.first();
    }

private:

    friend class detail::LayoutImpl;

    Field<T> reconstructField(FieldBase const & base) const { return Field<T>(base, _data.second()); }

    explicit Key(int nullOffset, int nullMask, int offset, Field<T> const & field) :
        _nullOffset(nullOffset), _nullMask(nullMask), _data(offset, field.getFieldData()) {}

    int _nullOffset;
    int _nullMask;
    boost::compressed_pair<int,typename Field<T>::FieldData> _data;
};

class LayoutBuilder {
public:

    template <typename T>
    Key<T> add(Field<T> const & field);

    Layout finish();

    LayoutBuilder();
    LayoutBuilder(LayoutBuilder const & other);
    
    LayoutBuilder & operator=(LayoutBuilder const & other);

    ~LayoutBuilder();

private:

    friend class Layout;

    class Impl;

    boost::shared_ptr<Impl> _impl;
};

class Layout {
public:

    template <typename T>
    struct Item {
        Field<T> field;
        Key<T> key;

        Item(Field<T> const & field_, Key<T> const & key_) : field(field_), key(key_) {}
    };

    typedef std::set<FieldDescription> Description;

    template <typename T>
    Item<T> find(Key<T> const & key) const;

    template <typename T>
    Item<T> find(std::string const & name) const;

    Description describe() const;

    ~Layout();

private:

    friend class LayoutBuilder;
    
    typedef detail::LayoutImpl Impl;

    Layout(boost::shared_ptr<Impl> const & impl);

    boost::shared_ptr<Impl> _impl;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Layout_h_INCLUDED
