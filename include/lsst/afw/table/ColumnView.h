// -*- lsst-c++ -*-
#ifndef AFW_TABLE_ColumnView_h_INCLUDED
#define AFW_TABLE_ColumnView_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/Layout.h"

namespace lsst { namespace afw { namespace table {

class TableBase;

class ColumnView {
public:
    
    Layout getLayout() const;
        
    template <typename T>
    typename ndarray::Array<T const,1> operator[](Key<T> const & key) const;

    template <typename T>
    typename ndarray::Array<T const,2,1> operator[](Key< Array<T> > const & key) const;

    ~ColumnView();

private:

    friend class TableBase;

    struct Impl;

    ColumnView(Layout const & layout, int recordCount, void * buf, ndarray::Manager::Ptr const & manager);

    boost::shared_ptr<Impl> _impl;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_ColumnView_h_INCLUDED
