// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_SchemaData_h_INCLUDED
#define AFW_TABLE_DETAIL_SchemaData_h_INCLUDED

#include <vector>
#include <algorithm>

#include "boost/variant.hpp"
#include "boost/mpl/transform.hpp"
#include "boost/type_traits/remove_const.hpp"
#include "boost/type_traits/remove_reference.hpp"

#include "lsst/afw/table/detail/RecordData.h"

namespace lsst { namespace afw { namespace table {

class Schema;

/**
 *  @brief A simple pair-like struct for mapping a Field (name and description) with a Key
 *         (used for actual data access).
 */
template <typename T>
struct SchemaItem {
    Key<T> key;
    Field<T> field;
};

namespace detail {

class Access;

/**
 *  @brief An internals class to hide the ugliness of the Schema implementation.
 *
 *  This can't be a full pImpl class, because some of the most important functionality
 *  is in the forEach function, a templated function we can't explicitly instantiate
 *  in a source file.  But putting all the details draws a clear line between what
 *  users should at (Schema) and what they shouldn't (this).
 *
 *  Because Schema holds SchemaData by shared pointer, one SchemaData can be shared between
 *  multiple Schemas, which use copy-on-write to create a new SchemaData if the pointer they have
 *  isn't unique.
 */
class SchemaData {
private:

    struct MakeItem {
        template <typename T>
        struct apply {
            typedef SchemaItem<T> type;
        };
    };

public:

    typedef boost::mpl::transform<FieldTypes,MakeItem>::type ItemTypes;
    typedef boost::make_variant_over<ItemTypes>::type ItemVariant;
    typedef std::vector<ItemVariant> ItemContainer;

    RecordId & getParentId(RecordData & record) const {
        return *reinterpret_cast<RecordId*>(&record + 1);
    }

    explicit SchemaData(bool hasParentId) :
        _recordSize(sizeof(RecordData)), _lastFlagField(-1), _lastFlagBit(-1),
        _hasParentId(hasParentId), _items()
    {
        if (hasParentId) _recordSize += sizeof(RecordId);
    }

private:

    template <typename F>
    struct VisitorWrapper : public boost::static_visitor<> {

        template <typename T>
        void operator()(SchemaItem<T> const & x) const { _func(x); };
    
        void operator()(ItemVariant const & v) const {
            boost::apply_visitor(*this, v);
        }

        explicit VisitorWrapper(F func) : _func(func) {}

    private:
        F _func;
    };

    friend class table::Schema;
    friend class detail::Access;

    int _recordSize;
    int _lastFlagField;
    int _lastFlagBit;
    bool _hasParentId;
    ItemContainer _items;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_SchemaData_h_INCLUDED
