// -*- c++ -*-
#ifndef CATALOG_ColumnView_h_INCLUDED
#define CATALOG_ColumnView_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "lsst/catalog/Layout.h"

namespace lsst { namespace catalog {

class SimpleTable;

class ColumnView {
public:
    
    Layout getLayout() const;
        
    template <typename T>
    typename ndarray::Array<T,1> operator[](Key<T> const & key) const;

    ~ColumnView();

private:

    friend class SimpleTable;

    struct Impl;

    ColumnView(Layout const & layout, int recordCount, char * buf, ndarray::Manager::Ptr const & manager);

    boost::shared_ptr<Impl> _impl;

    ndarray::detail::Core<1>::Ptr _intCore;
};

}} // namespace lsst::catalog

#endif // !CATALOG_ColumnView_h_INCLUDED
