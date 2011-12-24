// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_SchemaMapperImpl_h_INCLUDED
#define AFW_TABLE_DETAIL_SchemaMapperImpl_h_INCLUDED

#include <map>
#include <algorithm>

#include "boost/variant.hpp"
#include "boost/mpl/transform.hpp"
#include "boost/type_traits/remove_const.hpp"
#include "boost/type_traits/remove_reference.hpp"

#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw { namespace table {

class SchemaMapper;

namespace detail {

class SchemaMapperImpl {
private:

    struct MakeKeyPair {
        template <typename T>
        struct apply {
            typedef std::pair< Key<T>, Key<T> > type;
        };
    };

public:

    typedef boost::mpl::transform<FieldTypes,MakeKeyPair>::type KeyPairTypes;
    typedef boost::make_variant_over<KeyPairTypes>::type KeyPairVariant;
    typedef std::vector<KeyPairVariant> KeyPairMap;

    SchemaMapperImpl(Schema const & input, bool outputHasTree) : _input(input), _output(outputHasTree) {}

private:

    friend class table::SchemaMapper;
    friend class detail::Access;

    template <typename F>
    struct VisitorWrapper : public boost::static_visitor<> {

        template <typename T>
        void operator()(std::pair< Key<T>, Key<T> > const & pair) const {
            _func(pair.first, pair.second);
        }
        
        void operator()(KeyPairVariant const & v) const {
            boost::apply_visitor(*this, v);
        }

        explicit VisitorWrapper(F func) : _func(func) {}

    private:
        F _func;
    };

    Schema _input;
    Schema _output;
    KeyPairMap _map;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_SchemaMapperImpl_h_INCLUDED
