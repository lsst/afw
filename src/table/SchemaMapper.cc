#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw { namespace table {

namespace {

// Variant visitation functor used in SchemaMapper::invert()
struct SwapKeyPair : public boost::static_visitor<> {

    template <typename T>
    void operator()(std::pair< Key<T>, Key<T> > & pair) const {
        std::swap(pair.first, pair.second);
    }

    void operator()(detail::SchemaMapperImpl::KeyPairVariant & v) const {
        boost::apply_visitor(*this, v);
    }

};

// Variant visitation functor that returns true if the input key in a KeyPairVariant matches a
// the Key the functor was initialized with.
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

// Functor used to iterate through a minimal schema and map all fields present in the
// input schema and add those that are not.
struct MapMinimalSchema {

    template <typename U>
    void operator()(SchemaItem<U> const & item) const {
        Key<U> outputKey;
        if (_doMap) {
            try {
                SchemaItem<U> inputItem = _mapper->getInputSchema().find(item.key);
                outputKey = _mapper->addMapping(item.key);
            } catch (pex::exceptions::NotFoundError &) {
                outputKey = _mapper->addOutputField(item.field);
            }
        } else {
            outputKey = _mapper->addOutputField(item.field);
        }
        assert(outputKey == item.key);
    }

    explicit MapMinimalSchema(SchemaMapper * mapper, bool doMap) : _mapper(mapper), _doMap(doMap) {}

private:
    SchemaMapper * _mapper;
    bool _doMap;
};

// Schema::forEach functor that copies all fields from an schema to a schema mapper and maps them.
struct AddMapped {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        Field<T> field(prefix + item.field.getName(), item.field.getDoc(), item.field.getUnits(), item.field);
        mapper->addMapping(item.key, field);
    }

    explicit AddMapped(SchemaMapper * mapper_) : mapper(mapper_) {}

    SchemaMapper * mapper;
    std::string prefix;
};

// Schema::forEach functor that copies all fields from an schema to a schema mapper without mapping them.
struct AddUnmapped {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        Field<T> field(prefix + item.field.getName(), item.field.getDoc(), item.field.getUnits(), item.field);
        mapper->addOutputField(field);
    }

    explicit AddUnmapped(SchemaMapper * mapper_) : mapper(mapper_) {}

    SchemaMapper * mapper;
    std::string prefix;
};

struct RemoveMinimalSchema {

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        if (!minimal.contains(item)) {
            mapper->addMapping(item.key);
        }
    }

    RemoveMinimalSchema(SchemaMapper * mapper_, Schema const & minimal_) :
        mapper(mapper_), minimal(minimal_) {}

    SchemaMapper * mapper;
    Schema minimal;
};

} // anonymous

SchemaMapper::SchemaMapper() : _impl(new Impl(Schema(), Schema())) {}

SchemaMapper::SchemaMapper(SchemaMapper const & other) : _impl(new Impl(*other._impl)) {}

SchemaMapper::SchemaMapper(Schema const & input, Schema const & output) :
    _impl(new Impl(input, output))
{}

SchemaMapper::SchemaMapper(Schema const & input, bool shareAliasMap) :
    _impl(new Impl(input, Schema(input.getVersion())))
{
    if (shareAliasMap) {
        editOutputSchema().setAliasMap(input.getAliasMap());
    }
}

SchemaMapper & SchemaMapper::operator=(SchemaMapper const & other) {
    boost::scoped_ptr<Impl> tmp(new Impl(*other._impl));
    _impl.swap(tmp);
    return *this;
}

template <typename T>
Key<T> SchemaMapper::addMapping(Key<T> const & inputKey, bool doReplace) {
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
        Key<T> outputKey = _impl->_output.addField(inputField, doReplace);
        _impl->_map.insert(i, std::make_pair(inputKey, outputKey));
        return outputKey;
    }
}

template <typename T>
Key<T> SchemaMapper::addMapping(Key<T> const & inputKey, Field<T> const & field, bool doReplace) {
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
        Key<T> outputKey = _impl->_output.addField(field, doReplace);
        _impl->_map.insert(i, std::make_pair(inputKey, outputKey));
        return outputKey;
    }
}

template <typename T>
Key<T> SchemaMapper::addMapping(Key<T> const & inputKey, std::string const & outputName, bool doReplace) {
    typename Impl::KeyPairMap::iterator i = std::find_if(
        _impl->_map.begin(),
        _impl->_map.end(),
        KeyPairCompareEqual<T>(inputKey)
    );
    if (i != _impl->_map.end()) {
        Key<T> const & outputKey = boost::get< std::pair< Key<T>, Key<T> > >(*i).second;
        Field<T> field = _impl->_output.find(outputKey).field;
        field = field.copyRenamed(outputName);
        _impl->_output.replaceField(outputKey, field);
        return outputKey;
    } else {
        Field<T> inputField = _impl->_input.find(inputKey).field;
        Field<T> outputField = inputField.copyRenamed(outputName);
        Key<T> outputKey = _impl->_output.addField(outputField, doReplace);
        _impl->_map.insert(i, std::make_pair(inputKey, outputKey));
        return outputKey;
    }
}

void SchemaMapper::addMinimalSchema(Schema const & minimal, bool doMap) {
    if (getOutputSchema().getFieldCount() > 0) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicError,
            "Must add minimal schema to mapper before adding any other fields"
        );
    }
    MapMinimalSchema f(this, doMap);
    minimal.forEach(f);
}

SchemaMapper SchemaMapper::removeMinimalSchema(Schema const & input, Schema const & minimal) {
    SchemaMapper mapper(input);
    RemoveMinimalSchema f(&mapper, minimal);
    input.forEach(boost::ref(f));
    return mapper;
}

void SchemaMapper::invert() {
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
            lsst::pex::exceptions::NotFoundError,
            "Input Key is not mapped."
        );
    }
    return boost::get< std::pair< Key<T>, Key<T> > >(*i).second;
}

std::vector<SchemaMapper> SchemaMapper::join(
    std::vector<Schema> const & inputs,
    std::vector<std::string> const & prefixes
) {
    std::size_t const size = inputs.size();
    if (!prefixes.empty() && prefixes.size() != inputs.size()) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            (boost::format("prefix vector size (%d) must be the same as input vector size (%d)")
             % prefixes.size() % inputs.size()).str()
        );
    }
    std::vector<SchemaMapper> result;
    for (std::size_t i = 0; i < size; ++i) {
        result.push_back(SchemaMapper(inputs[i]));
    }
    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            if (i == j) {
                AddMapped functor(&result[j]);
                if (!prefixes.empty()) functor.prefix = prefixes[i];
                inputs[i].forEach(functor);
            } else {
                AddUnmapped functor(&result[j]);
                if (!prefixes.empty()) functor.prefix = prefixes[i];
                inputs[i].forEach(functor);
            }
        }
    }
    return result;
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUTMAPPER(r, data, elem)                         \
    template Key< elem > SchemaMapper::addOutputField(Field< elem > const &, bool); \
    template Key< elem > SchemaMapper::addMapping(Key< elem > const &, bool);       \
    template Key< elem > SchemaMapper::addMapping(Key< elem > const &, Field< elem > const &, bool); \
    template Key< elem > SchemaMapper::addMapping(Key< elem > const &, std::string const &, bool); \
    template bool SchemaMapper::isMapped(Key< elem > const &) const;    \
    template Key< elem > SchemaMapper::getMapping(Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUTMAPPER, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
