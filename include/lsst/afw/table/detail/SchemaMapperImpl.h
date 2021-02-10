// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_SchemaMapperImpl_h_INCLUDED
#define AFW_TABLE_DETAIL_SchemaMapperImpl_h_INCLUDED

#include <map>
#include <algorithm>

#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/types.h"
#include "lsst/afw/table/Schema.h"

namespace lsst {
namespace afw {
namespace table {

class SchemaMapper;

namespace detail {

/**
 * A private implementation class to hide the messy details of SchemaMapper.
 *
 * This class is very similar in spirit to SchemaImpl; look there for more information (though SchemaMapper
 * is not copy-on-write).
 */
class SchemaMapperImpl final {
private:
    /// Type metafunction that returns a std::variant of
    /// std::pair<Key<T>, Key<T>> given a list of the types for T.
    template <typename ...E>
    static std::variant<std::pair<Key<E>, Key<E>>...> makeKeyPairVariantType(TypeList<E...>);

public:
    /// A Variant type that can hold any one of the allowed pairx types.
    using KeyPairVariant = decltype(makeKeyPairVariantType(FieldTypes{}));
    /// A std::vector whose elements can be any of the allowed pair types.
    typedef std::vector<KeyPairVariant> KeyPairMap;

    /// Constructor from the given input and output schemas
    explicit SchemaMapperImpl(Schema const& input, Schema const& output) : _input(input), _output(output) {}

private:
    friend class table::SchemaMapper;
    friend class detail::Access;

    Schema _input;
    Schema _output;
    KeyPairMap _map;
};
}  // namespace detail
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_DETAIL_SchemaMapperImpl_h_INCLUDED
