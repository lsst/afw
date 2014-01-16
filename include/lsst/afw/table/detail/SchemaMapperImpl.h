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

#ifndef SWIG

namespace lsst { namespace afw { namespace table {

class SchemaMapper;

namespace detail {

/**
 *  @brief A private implementation class to hide the messy details of SchemaMapper.
 *
 *  This class is very similar in spirit to SchemaImpl, from the reason it's not a real
 *  pimpl (forEach) to Citizen; look there for more information (though SchemaMapper is
 *  not copy-on-write).
 */
class SchemaMapperImpl {
private:

    /// Boost.MPL metafunction that returns a std::pair< Key<T>, Key<T> > given a T.
    struct MakeKeyPair {
        template <typename T>
        struct apply {
            typedef std::pair< Key<T>, Key<T> > type;
        };
    };

public:

    /// An MPL sequence of all the allowed pair templates.
    typedef boost::mpl::transform<FieldTypes,MakeKeyPair>::type KeyPairTypes;
    /// A Boost.Variant type that can hold any one of the allowed pair types.
    typedef boost::make_variant_over<KeyPairTypes>::type KeyPairVariant;
        /// A std::vector whose elements can be any of the allowed pair types.
    typedef std::vector<KeyPairVariant> KeyPairMap;

    /// Constructor from the input schema; output schema is default-constructed.
    explicit SchemaMapperImpl(Schema const & input) : _input(input), _output() {}

    /**
     *  @brief A functor-wrapper used in the implementation of SchemaMapper::forEach.
     *
     *  See SchemaImpl::VisitorWrapper for discussion of the motivation.
     */
    template <typename F>
    struct VisitorWrapper : public boost::static_visitor<> {

        /// Call the wrapped function.
        template <typename T>
        void operator()(std::pair< Key<T>, Key<T> > const & pair) const {
            _func(pair.first, pair.second);
        }
        
        /**
         *  @brief Invoke the visitation.
         *
         *  The call to boost::apply_visitor will call the appropriate template of operator().
         *
         *  This overload allows a VisitorWrapper to be applied directly on a variant object
         *  with function-call syntax, allowing us to use it on our vector of variants with
         *  std::for_each and other STL algorithms.
         */
        void operator()(KeyPairVariant const & v) const {
            boost::apply_visitor(*this, v);
        }

        /// @brief Construct the wrappper.
        explicit VisitorWrapper(F func) : _func(func) {}

    private:
        F _func;
    };

private:

    friend class table::SchemaMapper;
    friend class detail::Access;

    Schema _input;
    Schema _output;
    KeyPairMap _map;
};

}}}} // namespace lsst::afw::table::detail

#endif // !SWIG

#endif // !AFW_TABLE_DETAIL_SchemaMapperImpl_h_INCLUDED
