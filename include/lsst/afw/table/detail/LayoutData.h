// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_LayoutData_h_INCLUDED
#define AFW_TABLE_DETAIL_LayoutData_h_INCLUDED

#include "lsst/afw/table/config.h"

#include <vector>
#include <algorithm>

#include "boost/variant.hpp"
#include "boost/mpl/transform.hpp"
#include "boost/type_traits/remove_const.hpp"
#include "boost/type_traits/remove_reference.hpp"

#include "lsst/afw/table/detail/RecordData.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief A simple pair-like struct for mapping a Field (name and description) with a Key
 *         (used for actual data access).
 */
template <typename T>
struct LayoutItem {
    Key<T> key;
    Field<T> field;
};

namespace detail {

struct LayoutData {

    static int const ALIGN_N_DOUBLE = 2;

    struct MakeItem {
        template <typename T>
        struct apply {
            typedef LayoutItem<T> type;
        };
    };

    typedef boost::mpl::transform<FieldTypes,MakeItem>::type ItemTypes;
    typedef boost::make_variant_over<ItemTypes>::type ItemVariant;
    typedef std::vector<ItemVariant> ItemContainer;

    template <
        typename F, 
        typename Result = typename boost::remove_const<
            typename boost::remove_reference<F>::type
            >::type::result_type
        >
    struct VisitorWrapper : public boost::static_visitor<Result> {

        typedef Result result_type;

        template <typename T>
        result_type operator()(LayoutItem<T> const & x) const { return _func(x); };
    
        result_type operator()(ItemVariant const & v) const {
            return boost::apply_visitor(*this, v);
        }

        explicit VisitorWrapper(F func = F()) : _func(func) {}

        F _func;
    };
    
    LayoutData() : recordSize(sizeof(RecordData)), items() {}

    template <typename F>
    void forEach(F func) const {
        VisitorWrapper<typename boost::unwrap_reference<F>::type &> visitor(func);
        std::for_each(items.begin(), items.end(), visitor);
    }

    int recordSize;
    ItemContainer items;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_LayoutData_h_INCLUDED
