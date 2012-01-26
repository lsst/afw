// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_Reader_h_INCLUDED
#define AFW_TABLE_IO_Reader_h_INCLUDED

#include "lsst/base.h"

namespace lsst { namespace afw { namespace table { 

class TableBase;

namespace io {

class Reader {
public:

    /// @brief Load an on-disk table into a container.
    template <typename ContainerT>
    ContainerT read() {
        Schema schema = _readSchema();
        PTR(typename ContainerT::Table) table 
            = boost::dynamic_pointer_cast<typename ContainerT::Table>(_readTable(schema));
        if (!table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Container's table type is not compatible with on-disk table type."
            );
        }
        ContainerT container(table);
        PTR(RecordBase) record = _readRecord(table);
        while (record) {
            container.insert(
                container.end(),
                boost::static_pointer_cast<typename ContainerT::Record>(record)
            );
            record = _readRecord(table);
        }
        return container;
    }
    
    virtual ~Reader() {}

protected:
    
    virtual Schema _readSchema(int nCols=-1) = 0;

    virtual PTR(TableBase) _readTable(Schema const & schema) = 0;

    virtual PTR(RecordBase) _readRecord(PTR(TableBase) const & table) = 0;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_Reader_h_INCLUDED
