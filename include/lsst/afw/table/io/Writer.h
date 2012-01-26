// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_Writer_h_INCLUDED
#define AFW_TABLE_IO_Writer_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/io/RecordSource.h"

namespace lsst { namespace afw { namespace table {

class TableBase;

namespace io {

class Writer {
public:

    /// @brief Load an on-disk table into a container.
    template <typename ContainerT>
    void write(ContainerT const & container) {
        detail::RecordSourceT<ContainerT> source(container);
        _write(container.getTable(), source);
    }
    
    virtual ~Writer() {}

protected:
    virtual void _write(CONST_PTR(TableBase) const & table, RecordSource & source) = 0;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_Writer_h_INCLUDED
