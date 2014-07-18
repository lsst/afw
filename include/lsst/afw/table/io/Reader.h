// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_Reader_h_INCLUDED
#define AFW_TABLE_IO_Reader_h_INCLUDED

#include "lsst/base.h"

namespace lsst { namespace afw { namespace table { 

class BaseTable;

namespace io {

/**
 *  @brief A base class for code that reads table/record data from another source.
 *
 *  An instance of Reader is associated with a particular file or other data source,
 *  and can be invoked simply by calling read() with no arguments and a template parameter
 *  that corresponds to the specialized container into which records will be loaded.
 *
 *  Reader provides driver code that should work for most input operations and record containers,
 *  and delegates the real work to its protected member functions.  Reader does not specify how it
 *  will be constructed.
 */ 
class Reader {
public:

    /**
     *  @brief Load an on-disk table into a container.
     *
     *  The container must be a specialized table container (like CatalogT):
     *   - It must be constructable from a single table shared_ptr argument.
     *   - It must have an insert member function that takes an position
     *     iterator and a record shared_ptr.
     */
    template <typename ContainerT>
    ContainerT read() {
#if 1
        // Work around a clang++ version 3.0 (tags/Apple/clang-211.12) bug with shared_ptr reference counts
        PTR(typename ContainerT::Table) table;
        {
            PTR(BaseTable) tmpTable = _readTable();
            table = boost::dynamic_pointer_cast<typename ContainerT::Table>(tmpTable);
        }
#else
        PTR(typename ContainerT::Table) table 
            = boost::dynamic_pointer_cast<typename ContainerT::Table>(_readTable());
#endif
        if (!table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError,
                "Container's table type is not compatible with on-disk table type."
            );
        }
        ContainerT container(table);
        PTR(BaseRecord) record = _readRecord(table);
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

    /**
     *  @brief Create a new table of the appropriate type.
     *
     *  The result may be an instance of a subclass of BaseTable.
     */
    virtual PTR(BaseTable) _readTable() = 0;

    /**
     *  @brief Read an individual record, creating it with the given table.
     *
     *  The result may be an instance of a subclass of BaseRecord.  The table will have just been loaded
     *  with _readSchema; these are separated in order to allow subclasses to delegate to base
     *  class implementations more effectively.
     */
    virtual PTR(BaseRecord) _readRecord(PTR(BaseTable) const & table) = 0;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_Reader_h_INCLUDED
