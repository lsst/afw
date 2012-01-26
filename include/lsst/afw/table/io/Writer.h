// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_Writer_h_INCLUDED
#define AFW_TABLE_IO_Writer_h_INCLUDED

#include "lsst/base.h"

namespace lsst { namespace afw { namespace table {

class TableBase;
class RecordBase;

namespace io {

class Writer {
public:

    /// @brief Load an on-disk table into a container.
    template <typename ContainerT>
    void write(ContainerT const & container) {
        _writeTable(container.getTable());
        for (typename ContainerT::const_iterator i = container.begin(); i != container.end(); ++i) {
            _writeRecord(*i);
        }
    }
    
    virtual ~Writer() {}

protected:
    virtual void _writeTable(CONST_PTR(TableBase) const & table) = 0;
    virtual void _writeRecord(RecordBase const & record) = 0;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_Writer_h_INCLUDED
