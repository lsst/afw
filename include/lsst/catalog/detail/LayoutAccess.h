// -*- c++ -*-
#ifndef CATALOG_DETAIL_LayoutAccess_h_INCLUDED
#define CATALOG_DETAIL_LayoutAccess_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/detail/LayoutData.h"

namespace lsst { namespace catalog { namespace detail {

struct LayoutAccess {

    static LayoutData const & getData(Layout const & layout) {
        return *layout._data;
    }

};

}}} // namespace lsst::catalog::detail

#endif // !CATALOG_DETAIL_LayoutAccess_h_INCLUDED
