#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/LayoutMapper.h"
#include "lsst/afw/table/RecordBase.h"

namespace lsst { namespace afw { namespace table {

namespace {

struct SwapKeyPair : public boost::static_visitor<> {

    template <typename T>
    void operator()(std::pair< Key<T>, Key<T> > & pair) const {
        std::swap(pair.first, pair.second);
    }

    void operator()(detail::LayoutMapperData::KeyPairVariant & v) const {
        boost::apply_visitor(*this, v);
    }

};

template <typename T>
struct KeyPairCompare : public boost::static_visitor<bool> {

    template <typename U>
    bool operator()(std::pair< Key<U>, Key<U> > const & pair) const {
        return _target == pair.first;
    }
    
    bool operator()(detail::LayoutMapperData::KeyPairVariant const & v) const {
        return boost::apply_visitor(*this, v);
    }

    KeyPairCompare(Key<T> const & target) : _target(target) {}

private:
    Key<T> const & _target;
};

struct CopyRecord {

    template <typename U>
    void operator()(Key<U> const & inputKey, Key<U> const & outputKey) const {
        detail::Access::copyValue(
            inputKey, _inputRecord,
            outputKey, _outputRecord
        );
    }

    CopyRecord(detail::RecordData * inputRecord, detail::RecordData * outputRecord) :
        _inputRecord(inputRecord), _outputRecord(outputRecord)
    {}

private:
    detail::RecordData * _inputRecord;
    detail::RecordData * _outputRecord;
};

} // anonymous

void LayoutMapper::_edit() {
    if (!_data.unique()) {
        boost::shared_ptr<Data> data(boost::make_shared<Data>(*_data));
        _data.swap(data);
    }
}

template <typename T>
Key<T> LayoutMapper::addMapping(Key<T> const & inputKey) {
    _edit();
    typename Data::KeyPairMap::iterator i = std::find_if(
        _data->_map.begin(),
        _data->_map.end(),
        KeyPairCompare<T>(inputKey)
    );
    Field<T> inputField = _data->_input.find(inputKey).field;
    if (i != _data->_map.end()) {
        Key<T> const & outputKey = boost::get< std::pair< Key<T>, Key<T> > >(*i).second;
        _data->_output.replaceField(outputKey, inputField);
        return outputKey;
    } else {
        Key<T> outputKey = _data->_output.addField(inputField);
        _data->_map.insert(i, std::make_pair(inputKey, outputKey));
        return outputKey;
    }
}

template <typename T>
Key<T> LayoutMapper::addMapping(Key<T> const & inputKey, Field<T> const & field) {
    _edit();
    typename Data::KeyPairMap::iterator i = std::find_if(
        _data->_map.begin(),
        _data->_map.end(),
        KeyPairCompare<T>(inputKey)
    );
    if (i != _data->_map.end()) {
        Key<T> const & outputKey = boost::get< std::pair< Key<T>, Key<T> > >(*i).second;
        _data->_output.replaceField(outputKey, field);
        return outputKey;
    } else {
        Key<T> outputKey = _data->_output.addField(field);
        _data->_map.insert(i, std::make_pair(inputKey, outputKey));
        return outputKey;
    }
}

void LayoutMapper::invert() {
    _edit();
    std::swap(_data->_input, _data->_output);
    std::for_each(_data->_map.begin(), _data->_map.end(), SwapKeyPair());
}

template <typename T>
bool LayoutMapper::isMapped(Key<T> const & inputKey) const {
    return std::count_if(
        _data->_map.begin(),
        _data->_map.end(),
        KeyPairCompare<T>(inputKey)
    );
}

template <typename T>
Key<T> LayoutMapper::getMapping(Key<T> const & inputKey) const {
    typename Data::KeyPairMap::iterator i = std::find_if(
        _data->_map.begin(),
        _data->_map.end(),
        KeyPairCompare<T>(inputKey)
    );
    if (i == _data->_map.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            "Input Key is not mapped."
        );
    }
    return boost::get< std::pair< Key<T>, Key<T> > >(*i).second;
}

void LayoutMapper::copyRecord(RecordBase const & input, RecordBase const & output) const {
    this->forEach(CopyRecord(input._data, output._data));
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUTMAPPER(r, data, elem)                         \
    template Key< elem > LayoutMapper::addOutputField(Field< elem > const &);      \
    template Key< elem > LayoutMapper::addMapping(Key< elem > const &);       \
    template Key< elem > LayoutMapper::addMapping(Key< elem > const &, Field< elem > const &); \
    template bool LayoutMapper::isMapped(Key< elem > const &) const;    \
    template Key< elem > LayoutMapper::getMapping(Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUTMAPPER, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
