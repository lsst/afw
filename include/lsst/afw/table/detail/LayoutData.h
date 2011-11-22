// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_LayoutData_h_INCLUDED
#define AFW_TABLE_DETAIL_LayoutData_h_INCLUDED

#include "lsst/afw/table/detail/fusion_limits.h"

#include <vector>

#include "boost/mpl/transform.hpp"
#include "boost/fusion/algorithm/iteration/for_each.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"

#include "lsst/afw/table/Layout.h"

namespace lsst { namespace afw { namespace table { namespace detail {

class RecordAux {
public:
    typedef boost::shared_ptr<RecordAux> Ptr;
    virtual ~RecordAux() {}
};

struct RecordData {
    typedef boost::uint64_t IdType;

    IdType id;
    RecordAux::Ptr aux;
    RecordData * parent;
    RecordData * child;
    RecordData * sibling;

    RecordData() : id(0), aux(), parent(0), child(0), sibling(0) {}
};

struct LayoutData {

    static int const ALIGN_N_DOUBLE = 2;

    struct MakeItemVectorPair {
        template <typename T> struct apply {
            typedef boost::fusion::pair< T, std::vector< Layout::Item<T> > > type;
        };
    };

    template <typename Function>
    struct IterateItemVector {

        template <typename T>
        void operator()(boost::fusion::pair< T, std::vector< Layout::Item<T> > > const & type) const {
            for (
                typename std::vector< Layout::Item<T> >::const_iterator i = type.second.begin();
                i != type.second.end();
                ++i
            ) {
                func(*i);
            }
        };
        
        explicit IterateItemVector(Function func_) : func(func_) {}

        Function func;
    };

    typedef boost::fusion::result_of::as_map<
        boost::mpl::transform< detail::FieldTypes, MakeItemVectorPair >::type
        >::type ItemContainer;

    LayoutData() : recordSize(sizeof(RecordData)), items() {}

    template <typename Function>
    void forEachItem(Function func) const {
        IterateItemVector<Function> metaFunc(func);
        boost::fusion::for_each(items, metaFunc);
    }

    int recordSize;
    ItemContainer items;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_LayoutData_h_INCLUDED
