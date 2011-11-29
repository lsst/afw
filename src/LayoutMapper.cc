#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/LayoutMapper.h"

namespace lsst { namespace afw { namespace table {

namespace {

struct InvertMap {

    template <typename T>
    void operator()(boost::fusion::pair< T, std::map< Key<T>, Key<T> > > & pair) const {
        typedef std::map< Key<T>, Key<T> > Map;
        Map inverted;
        for (typename Map::iterator i = pair.second.begin(); i != pair.second.end(); ++i) {
            inverted.insert(std::make_pair(i->second, i->first));
        }
        pair.second.swap(inverted);
    }

};

} // anonymous

template <typename T>
Key<T> LayoutMapper::copy(Key<T> const & inputKey) {
    typedef std::map< Key<T>, Key<T> > Map;
    Map & map = boost::fusion::at_key<T>(_maps);
    typename Map::iterator i = map.lower_bound(inputKey);
    Field<T> inputField = _input.find(inputKey).field;
    if (i != map.end() && i->first == inputKey) {
        _output.replace(i->second, inputField);
        return i->second;
    }
    Key<T> outputKey = _output.add(inputField);
    map.insert(i, std::make_pair(inputKey, outputKey));
    return outputKey;
}

template <typename T>
Key<T> LayoutMapper::copy(Key<T> const & inputKey, Field<T> const & field) {
    typedef std::map< Key<T>, Key<T> > Map;
    Map & map = boost::fusion::at_key<T>(_maps);
    typename Map::iterator i = map.lower_bound(inputKey);
    if (i != map.end() && i->first == inputKey) {
        _output.replace(i->second, field);
        return i->second;
    }
    Key<T> outputKey = _output.add(field);
    map.insert(i, std::make_pair(inputKey, outputKey));
    return outputKey;
}

void LayoutMapper::invert() {
    std::swap(_input, _output);
    boost::fusion::for_each(_maps, InvertMap());
}

template <typename T>
bool LayoutMapper::isMapped(Key<T> const & inputKey) const {
    typedef std::map< Key<T>, Key<T> > Map;
    Map const & map = boost::fusion::at_key<T>(_maps);
    return map.count(inputKey);
}

template <typename T>
Key<T> LayoutMapper::getMapping(Key<T> const & inputKey) const {
    typedef std::map< Key<T>, Key<T> > Map;
    Map const & map = boost::fusion::at_key<T>(_maps);
    typename Map::const_iterator i = map.find(inputKey);
    if (i == map.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            "Input Key is not mapped."
        );
    }
    return i->second;
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUTMAPPER(r, data, elem)                         \
    template Key< elem > LayoutMapper::add(Field< elem > const &);      \
    template Key< elem > LayoutMapper::copy(Key< elem > const &);       \
    template Key< elem > LayoutMapper::copy(Key< elem > const &, Field< elem > const &); \
    template bool LayoutMapper::isMapped(Key< elem > const &) const;    \
    template Key< elem > LayoutMapper::getMapping(Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUTMAPPER, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
