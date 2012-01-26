// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_Reader_h_INCLUDED
#define AFW_TABLE_IO_Reader_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/io/RecordSink.h"

namespace lsst { namespace afw { namespace table { 

class TableBase;

namespace io {

class Reader {
public:

    /// @brief Load an on-disk table into a container.
    template <typename ContainerT>
    ContainerT read() {
        PTR(typename ContainerT::Table) table 
            = boost::dynamic_pointer_cast<typename ContainerT::Table>(_readTable());
        if (!table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Container's table type is not compatible with on-disk table type."
            );
        }
        ContainerT container(table);
        detail::RecordSinkT<ContainerT> sink(container);
        _readRecords(table, sink);
        return container;
    }
    
    virtual ~Reader() {}

protected:
    
    virtual PTR(TableBase) _readTable() = 0;

    virtual void _readRecords(PTR(TableBase) const & table, RecordSink & sink) = 0;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_Reader_h_INCLUDED
