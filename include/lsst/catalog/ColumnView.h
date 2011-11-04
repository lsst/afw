// -*- c++ -*-
#ifndef CATALOG_ColumnView_h_INCLUDED
#define CATALOG_ColumnView_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include "lsst/catalog/Layout.h"

namespace lsst { namespace catalog {

class TableBase;

class ColumnView {
public:
    
    typedef ndarray::detail::UnaryOpExpression< 
        ndarray::Array<int,1>,
        ndarray::detail::BitwiseAndTag::ExprScalar< ndarray::Array<int,1>, int >::Bound
    > IsNullColumn;

    Layout getLayout() const { return _layout; }

    template <typename T>
    IsNullColumn isNull(Key<T> const & key) const;
        
    template <typename T>
    typename Field<T>::Column operator[](Key<T> const & key) const;

private:

    friend class TableBase;

    ColumnView(Layout const & layout, int recordCount, char * buf, ndarray::Manager::Ptr const & manager);

    int _recordCount;
    char * _buf;
    Layout _layout;
    ndarray::detail::Core<1>::Ptr _intCore;
};

}} // namespace lsst::catalog

#endif // !CATALOG_ColumnView_h_INCLUDED
