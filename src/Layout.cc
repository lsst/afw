
#define FUSION_MAX_VECTOR_SIZE 20
#define FUSION_MAX_MAP_SIZE 20

#include "boost/mpl/transform.hpp"
#include "boost/fusion/algorithm/iteration/for_each.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/catalog/Layout.h"

namespace lsst { namespace catalog {

namespace {

template <typename T>
struct InternalItem {
    FieldBase field;
    Key<T> key;
};

struct MakeInternalItemPair {
    template <typename T> struct apply {
        typedef boost::fusion::pair< T, std::vector< InternalItem<T> > > type;
    };
};

typedef boost::fusion::result_of::as_map<
    boost::mpl::transform< detail::FieldTypes, MakeInternalItemPair >::type
    >::type LayoutData;

struct LayoutGap {
    int offset;
    int size;
};

} // anonymous

namespace detail {

//----- Layout private implementation -----------------------------------------------------------------------

class LayoutImpl {
public:

    template <typename T>
    static Layout::Item<T> makePublicItem(InternalItem<T> const & internal) {
        return Layout::Item<T>(internal.key.reconstructField(internal.field), internal.key);
    }

    template <typename T>
    static FieldDescription describe(InternalItem<T> const & internal) {
        return internal.key.reconstructField(internal.field).describe();
    }

    template <typename T>
    static Key<T> makeKey(int nullOffset, int nullMask, int offset, Field<T> const & field) {
        return Key<T>(nullOffset, nullMask, offset, field);
    }

    LayoutImpl() : _recordSize(0), _data() {}

    int _recordSize;
    LayoutData _data;
};

} // namespace detail

//----- LayoutBuilder private implementation ----------------------------------------------------------------

class LayoutBuilder::Impl {
public:

    typedef detail::LayoutImpl Result;

    template <typename T>
    Key<T> add(Field<T> const & field);

    boost::shared_ptr<Result> finish();

    Impl() : _result(new Result()), _gaps(), _currentNullOffset(0), _currentNullMask(0) {}

    Impl(Impl const & other) :
        _result(new Result(*other._result)), _gaps(other._gaps), 
        _currentNullOffset(other._currentNullOffset),
        _currentNullMask(other._currentNullMask)
    {}

private:

    void operator=(Impl const & other);

    int findOffset(int size, int align);

    boost::shared_ptr<Result> _result;
    std::list<LayoutGap> _gaps;
    int _currentNullOffset;
    int _currentNullMask;
};

template <typename T>
Key<T> LayoutBuilder::Impl::add(Field<T> const & field) {
    if (!_result.unique()) {
        boost::shared_ptr<Result> result(new Result(*_result));
        _result.swap(result);
    }
    if (!_currentNullMask) {
        _currentNullOffset = findOffset(sizeof(int), sizeof(int));
        _currentNullMask = 1;
    }
    int offset = findOffset(field.getByteSize(), field.getByteAlign());
    InternalItem<T> i = { 
        field,
        Result::makeKey(_currentNullOffset, _currentNullMask, offset, field)
    };
    _currentNullMask <<= 1;
    boost::fusion::at_key<T>(_result->_data).push_back(i);
    return i.key;
}

boost::shared_ptr<LayoutBuilder::Impl::Result> LayoutBuilder::Impl::finish() {
    static int const MIN_RECORD_ALIGN = sizeof(double) * 2;
    _result->_recordSize += (MIN_RECORD_ALIGN - _result->_recordSize % MIN_RECORD_ALIGN);
    return _result;
}

int LayoutBuilder::Impl::findOffset(int size, int align) {
    for (std::list<LayoutGap>::iterator i = _gaps.begin(); i != _gaps.end(); ++i) {
        if (i->offset % align == 0 && i->size >= size) {
            int offset = i->offset;
            if (i->size == size) {
                _gaps.erase(i);
            } else {
                i->offset += size;
                i->size -= size;
            }
            return offset;
        }
    }
    int extra = align - _result->_recordSize % align;
    if (extra == align) {
        int offset = _result->_recordSize;
        _result->_recordSize += size;
        return offset;
    } else {
        LayoutGap gap = { _result->_recordSize, extra };
        _result->_recordSize += extra;
        _gaps.push_back(gap);
        int offset = _result->_recordSize;
        _result->_recordSize += size;
        return offset;
    }
}

//----- LayoutBuilder public implementation -----------------------------------------------------------------

template <typename T>
Key<T> LayoutBuilder::add(Field<T> const & field) {
    if (!_impl.unique()) {
        boost::shared_ptr<Impl> impl(new Impl(*_impl));
        _impl.swap(impl);
    }
    return _impl->add(field);
}

Layout LayoutBuilder::finish() {
    return Layout(_impl->finish());
}

LayoutBuilder::LayoutBuilder() : _impl(new Impl()) {}

LayoutBuilder::LayoutBuilder(LayoutBuilder const & other) : _impl(other._impl) {}

LayoutBuilder & LayoutBuilder::operator=(LayoutBuilder const & other) {
    _impl = other._impl;
    return *this;
}

LayoutBuilder::~LayoutBuilder() {}

//----- Layout public implementation ------------------------------------------------------------------------

template <typename T>
Layout::Item<T> Layout::find(Key<T> const & key) const {
    std::vector< InternalItem<T> > const & vec = boost::fusion::at_key<T>(_impl->_data);
    for (typename std::vector< InternalItem<T> >::const_iterator i = vec.begin(); i != vec.end(); ++i) {
        if (i->key == key) return Impl::makePublicItem(*i);
    }
    throw std::invalid_argument("Key not found.");
}

template <typename T>
Layout::Item<T> Layout::find(std::string const & name) const {
    std::vector< InternalItem<T> > const & vec = boost::fusion::at_key<T>(_impl->_data);
    for (typename std::vector< InternalItem<T> >::const_iterator i = vec.begin(); i != vec.end(); ++i) {
        if (i->field.name == name) return Impl::makePublicItem(*i);
    }
    throw std::invalid_argument("Name not found.");        
}

namespace {

struct Describe {

    template <typename T>
    void operator()(boost::fusion::pair< T, std::vector< InternalItem<T> > > const & type) const {
        for (
             typename std::vector< InternalItem<T> >::const_iterator i = type.second.begin();
             i != type.second.end();
             ++i
        ) {
            result->insert(detail::LayoutImpl::describe(*i));
        }
    }

    Layout::Description * result;
};

} // anonymous

Layout::Description Layout::describe() const {
    Description result;
    Describe f = { &result };
    boost::fusion::for_each(_impl->_data, f);
    return result;
}

int Layout::getRecordSize() const {
    return _impl->_recordSize;
}

Layout::Layout(boost::shared_ptr<Layout::Impl> const & impl) : _impl(impl) {}

Layout::~Layout() {}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUT(r, data, elem)                           \
    template Key< elem > LayoutBuilder::add(Field< elem > const &); \
    template Layout::Item< elem > Layout::find(Key< elem > const &) const; \
    template Layout::Item< elem > Layout::find(std::string const & ) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUT, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog
