// -*- c++ -*-
#ifndef CATALOG_Layout_h_INCLUDED
#define CATALOG_Layout_h_INCLUDED

#include <vector>
#include <list>
#include <set>
#include <string>
#include <stdexcept>

#define FUSION_MAX_VECTOR_SIZE 20
#define FUSION_MAX_MAP_SIZE 20

#include "boost/mpl/transform.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"
#include "boost/compressed_pair.hpp"
#include "boost/type_traits/is_same.hpp"

#include "lsst/ndarray.h"
#include "lsst/catalog/Field.h"

namespace lsst { namespace catalog {

class Layout;
class Table;

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

    friend class Layout;
    friend class Table;

    typename Field<T>::Column 
    makeColumn(void * buf, int recordCount, ndarray::Manager::Ptr const & manager) const {
        return Field<T>::makeColumn(
            reinterpret_cast<char *>(buf) + _data.first() * recordCount,
            recordCount,
            manager,
            _data.second()
        );
    }

    explicit Key(int offset, Field<T> const & field) : _data(offset, field.getFieldData()) {}

    boost::compressed_pair<int,typename Field<T>::FieldData> _data;
};

class Layout {
public:

    template <typename T>
    struct Item {
        Field<T> field;
        Key<T> key;

        Item(Field<T> const & field_, Key<T> const & key_) : field(field_), key(key_) {}
    };

private:

    struct MakeItemPair {
        template <typename T> struct apply {
            typedef boost::fusion::pair< T, std::vector< Item<T> > > type;
        };
    };

    struct Describe;

    typedef boost::fusion::result_of::as_map<
        boost::mpl::transform< detail::FieldTypes, MakeItemPair >::type
    >::type Data;

    struct Gap {
        int offset;
        int size;
    };

    static int const MIN_RECORD_ALIGN = sizeof(double) * 2;

public:

    typedef std::set<FieldDescription> Description;

    template <typename T>
    Key<T> addField(Field<T> const & field) {
        int offset = findOffset(field.getByteSize(), field.getByteAlign());
        Item<T> i(field, Key<T>(offset, field));
        boost::fusion::at_key<T>(_data).push_back(i);
        return i.key;
    }

    template <typename T>
    Item<T> find(Key<T> const & key) const {
        std::vector< Item<T> > const & vec = boost::fusion::at_key<T>(_data);
        for (typename std::vector< Item<T> >::const_iterator i = vec.begin(); i != vec.end(); ++i) {
            if (i->key == key) return *i;
        }
        throw std::invalid_argument("Key not found.");
    }

    template <typename T>
    Item<T> find(std::string const & name) const {
        std::vector< Item<T> > const & vec = boost::fusion::at_key<T>(_data);
        for (typename std::vector< Item<T> >::const_iterator i = vec.begin(); i != vec.end(); ++i) {
            if (i->field.name == name) return *i;
        }
        throw std::invalid_argument("Name not found.");        
    }

    Description describe() const;

    int computeStride() const { return _bytes + (MIN_RECORD_ALIGN - _bytes % MIN_RECORD_ALIGN); }

private:

    int findOffset(int size, int align);

    Data _data;
    int _bytes;
    std::list<Gap> _gaps;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Layout_h_INCLUDED
