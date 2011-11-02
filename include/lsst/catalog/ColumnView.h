// -*- c++ -*-
#ifndef CATALOG_ColumnView_h_INCLUDED
#define CATALOG_ColumnView_h_INCLUDED

#include "lsst/catalog/Layout.h"

namespace lsst { namespace catalog {

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
    int _recordCount;
    ndarray::Manager::Ptr _manager;
    void * _buf;
    Layout _layout;
};

}} // namespace lsst::catalog

#endif // !CATALOG_ColumnView_h_INCLUDED
