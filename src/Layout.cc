#include <vector>
#include <list>
#include <stdexcept>

#define FUSION_MAX_VECTOR_SIZE 20
#define FUSION_MAX_MAP_SIZE 20

#include "boost/make_shared.hpp"
#include "boost/mpl/transform.hpp"
#include "boost/fusion/algorithm/iteration/for_each.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/detail/KeyAccess.h"

namespace lsst { namespace catalog {

namespace {

struct MakeKeyVectorPair {
    template <typename T> struct apply {
        typedef boost::fusion::pair< T, std::vector< Key<T> > > type;
    };
};

typedef boost::fusion::result_of::as_map<
    boost::mpl::transform< detail::FieldTypes, MakeKeyVectorPair >::type
    >::type KeyContainer;

struct LayoutGap {
    int offset;
    int size;
};

} // anonymous

//----- Layout private implementation -----------------------------------------------------------------------

struct Layout::Data {

    Data() : recordSize(0), keys() {}

    int recordSize;
    KeyContainer keys;
};

//----- LayoutBuilder private implementation ----------------------------------------------------------------

class LayoutBuilder::Impl {
public:

    typedef Layout::Data Data;

    template <typename T>
    Key<T> add(Field<T> const & field);

    boost::shared_ptr<Data> finish();

    Impl() : _data(new Data()), _gaps(), _currentNullOffset(0), _currentNullMask(0) {}

    Impl(Impl const & other) :
        _data(new Data(*other._data)), _gaps(other._gaps), 
        _currentNullOffset(other._currentNullOffset),
        _currentNullMask(other._currentNullMask)
    {}

private:

    void operator=(Impl const & other);

    int findOffset(int size, int align);

    boost::shared_ptr<Data> _data;
    std::list<LayoutGap> _gaps;
    int _currentNullOffset;
    int _currentNullMask;
};

template <typename T>
Key<T> LayoutBuilder::Impl::add(Field<T> const & field) {
    if (!_data.unique()) {
        boost::shared_ptr<Data> result(new Data(*_data));
        _data.swap(result);
    }
    if (!_currentNullMask) {
        _currentNullOffset = findOffset(sizeof(int), sizeof(int));
        _currentNullMask = 1;
    }
    boost::shared_ptr< detail::KeyData<T> > keyData = boost::make_shared< detail::KeyData<T> >(field);
    keyData->offset = findOffset(field.getByteSize(), field.getByteAlign());
    keyData->nullOffset = _currentNullOffset;
    keyData->nullMask = _currentNullMask;
    Key<T> key = detail::KeyAccess::make(keyData);
    _currentNullMask <<= 1;
    boost::fusion::at_key<T>(_data->keys).push_back(key);
    return key;
}

boost::shared_ptr<LayoutBuilder::Impl::Data> LayoutBuilder::Impl::finish() {
    static int const MIN_RECORD_ALIGN = sizeof(double) * 2;
    _data->recordSize += (MIN_RECORD_ALIGN - _data->recordSize % MIN_RECORD_ALIGN);
    return _data;
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
    int extra = align - _data->recordSize % align;
    if (extra == align) {
        int offset = _data->recordSize;
        _data->recordSize += size;
        return offset;
    } else {
        LayoutGap gap = { _data->recordSize, extra };
        _data->recordSize += extra;
        _gaps.push_back(gap);
        int offset = _data->recordSize;
        _data->recordSize += size;
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
Key<T> Layout::find(std::string const & name) const {
    std::vector< Key<T> > const & vec = boost::fusion::at_key<T>(_data->keys);
    for (typename std::vector< Key<T> >::const_iterator i = vec.begin(); i != vec.end(); ++i) {
        if (i->getField().name == name) return *i;
    }
    throw std::invalid_argument("Name not found.");        
}

namespace {

struct Describe {

    template <typename T>
    void operator()(boost::fusion::pair< T, std::vector< Key<T> > > const & type) const {
        for (
             typename std::vector< Key<T> >::const_iterator i = type.second.begin();
             i != type.second.end();
             ++i
        ) {
            result->insert(i->getField().describe());
        }
    }

    Layout::Description * result;
};

} // anonymous

Layout::Description Layout::describe() const {
    Description result;
    Describe f = { &result };
    boost::fusion::for_each(_data->keys, f);
    return result;
}

int Layout::getRecordSize() const {
    return _data->recordSize;
}

Layout::Layout(boost::shared_ptr<Layout::Data> const & data) : _data(data) {}

Layout::~Layout() {}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUT(r, data, elem)                           \
    template Key< elem > LayoutBuilder::add(Field< elem > const &); \
    template Key< elem > Layout::find(std::string const & ) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUT, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog
