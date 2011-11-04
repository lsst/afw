// -*- c++ -*-
#ifndef CATALOG_DETAIL_LayoutData_h_INCLUDED
#define CATALOG_DETAIL_LayoutData_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include <vector>

#include "boost/mpl/transform.hpp"
#include "boost/fusion/algorithm/iteration/for_each.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"

#include "lsst/catalog/Layout.h"

namespace lsst { namespace catalog { namespace detail {

struct LayoutData {

    struct MakeKeyVectorPair {
        template <typename T> struct apply {
            typedef boost::fusion::pair< T, std::vector< Key<T> > > type;
        };
    };

    template <typename Function>
    struct IterateKeyVector {

        template <typename T>
        void operator()(boost::fusion::pair< T, std::vector< Key<T> > > const & type) const {
            for (
                typename std::vector< Key<T> >::const_iterator i = type.second.begin();
                i != type.second.end();
                ++i
            ) {
                func(*i);
            }
        };
        
        explicit IterateKeyVector(Function func_) : func(func_) {}

        Function func;
    };

    typedef boost::fusion::result_of::as_map<
        boost::mpl::transform< detail::FieldTypes, MakeKeyVectorPair >::type
        >::type KeyContainer;

    LayoutData() : recordSize(0), keys() {}

    template <typename Function>
    void forEachKey(Function func) const {
        IterateKeyVector<Function> metaFunc(func);
        boost::fusion::for_each(keys, metaFunc);
    }

    int recordSize;
    KeyContainer keys;
};

}}} // namespace lsst::catalog::detail

#endif // !CATALOG_DETAIL_LayoutData_h_INCLUDED
