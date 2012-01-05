#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/RecordBase.h"

namespace lsst { namespace afw { namespace table {

namespace {

struct SwapKeyPair : public boost::static_visitor<> {

    template <typename T>
    void operator()(std::pair< Key<T>, Key<T> > & pair) const {
        std::swap(pair.first, pair.second);
    }

    void operator()(detail::SchemaMapperImpl::KeyPairVariant & v) const {
        boost::apply_visitor(*this, v);
    }

};

template <typename T>
struct KeyPairCompareEqual : public boost::static_visitor<bool> {

    template <typename U>
    bool operator()(std::pair< Key<U>, Key<U> > const & pair) const {
        return _target == pair.first;
    }
    
    bool operator()(detail::SchemaMapperImpl::KeyPairVariant const & v) const {
        return boost::apply_visitor(*this, v);
    }

    KeyPairCompareEqual(Key<T> const & target) : _target(target) {}

private:
    Key<T> const & _target;
};

struct KeyPairCompareFirst : public boost::static_visitor<int> {

    template <typename U>
    int operator()(std::pair< Key<U>, Key<U> > const & pair) const {
        return pair.first.getOffset();
    }
    
    bool operator()(
        detail::SchemaMapperImpl::KeyPairVariant const & a,
        detail::SchemaMapperImpl::KeyPairVariant const & b
    ) const {
        return boost::apply_visitor(*this, a) < boost::apply_visitor(*this, b);
    }

};

struct KeyPairCompareSecond: public boost::static_visitor<int> {

    template <typename U>
    int operator()(std::pair< Key<U>, Key<U> > const & pair) const {
        return pair.second.getOffset();
    }
    
    bool operator()(
        detail::SchemaMapperImpl::KeyPairVariant const & a,
        detail::SchemaMapperImpl::KeyPairVariant const & b
    ) const {
        return boost::apply_visitor(*this, a) < boost::apply_visitor(*this, b);
    }

};

} // anonymous

SchemaMapper::SchemaMapper(Schema const & input, bool outputHasTree) :
    _impl(boost::make_shared<Impl>(input, outputHasTree))
{}

void SchemaMapper::_edit() {
    if (!_impl.unique()) {
        boost::shared_ptr<Impl> impl(boost::make_shared<Impl>(*_impl));
        _impl.swap(impl);
    }
}

template <typename T>
Key<T> SchemaMapper::addMapping(Key<T> const & inputKey) {
    _edit();
    typename Impl::KeyPairMap::iterator i = std::find_if(
        _impl->_map.begin(),
        _impl->_map.end(),
        KeyPairCompareEqual<T>(inputKey)
    );
    Field<T> inputField = _impl->_input.find(inputKey).field;
    if (i != _impl->_map.end()) {
        Key<T> const & outputKey = boost::get< std::pair< Key<T>, Key<T> > >(*i).second;
        _impl->_output.replaceField(outputKey, inputField);
        return outputKey;
    } else {
        Key<T> outputKey = _impl->_output.addField(inputField);
        _impl->_map.insert(i, std::make_pair(inputKey, outputKey));
        return outputKey;
    }
}

template <typename T>
Key<T> SchemaMapper::addMapping(Key<T> const & inputKey, Field<T> const & field) {
    _edit();
    typename Impl::KeyPairMap::iterator i = std::find_if(
        _impl->_map.begin(),
        _impl->_map.end(),
        KeyPairCompareEqual<T>(inputKey)
    );
    if (i != _impl->_map.end()) {
        Key<T> const & outputKey = boost::get< std::pair< Key<T>, Key<T> > >(*i).second;
        _impl->_output.replaceField(outputKey, field);
        return outputKey;
    } else {
        Key<T> outputKey = _impl->_output.addField(field);
        _impl->_map.insert(i, std::make_pair(inputKey, outputKey));
        return outputKey;
    }
}

void SchemaMapper::invert() {
    _edit();
    std::swap(_impl->_input, _impl->_output);
    std::for_each(_impl->_map.begin(), _impl->_map.end(), SwapKeyPair());
}

template <typename T>
bool SchemaMapper::isMapped(Key<T> const & inputKey) const {
    return std::count_if(
        _impl->_map.begin(),
        _impl->_map.end(),
        KeyPairCompareEqual<T>(inputKey)
    );
}

template <typename T>
Key<T> SchemaMapper::getMapping(Key<T> const & inputKey) const {
    typename Impl::KeyPairMap::iterator i = std::find_if(
        _impl->_map.begin(),
        _impl->_map.end(),
        KeyPairCompareEqual<T>(inputKey)
    );
    if (i == _impl->_map.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            "Input Key is not mapped."
        );
    }
    return boost::get< std::pair< Key<T>, Key<T> > >(*i).second;
}

void SchemaMapper::sort(SortOrder order) {
    _edit();
    switch (order) {
    case INPUT:
        std::sort(_impl->_map.begin(), _impl->_map.end(), KeyPairCompareFirst());
        break;
    case OUTPUT:
        std::sort(_impl->_map.begin(), _impl->_map.end(), KeyPairCompareSecond());
        break;
    }
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUTMAPPER(r, data, elem)                         \
    template Key< elem > SchemaMapper::addOutputField(Field< elem > const &);      \
    template Key< elem > SchemaMapper::addMapping(Key< elem > const &);       \
    template Key< elem > SchemaMapper::addMapping(Key< elem > const &, Field< elem > const &); \
    template bool SchemaMapper::isMapped(Key< elem > const &) const;    \
    template Key< elem > SchemaMapper::getMapping(Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUTMAPPER, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
