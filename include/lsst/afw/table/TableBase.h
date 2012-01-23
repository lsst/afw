// -*- lsst-c++ -*-
#ifndef AFW_TABLE_TableBase_h_INCLUDED
#define AFW_TABLE_TableBase_h_INCLUDED

#include "lsst/base.h"
#include "lsst/ndarray/Manager.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/IdFactory.h"

namespace lsst { namespace afw { namespace table {

class TableBase {
public:

    /// @brief Number of records in each block when capacity is not given explicitly.
    static int nRecordsPerBlock;

    /**
     *  @brief Return a polymorphic deep copy.
     *
     *  Derived classes should reimplement by static-casting the output of _clone to a
     *  pointer-to-derived.
     */
    PTR(TableBase) clone() const { return _clone(); }
    
    /// @brief Return the table's schema.
    Schema const & getSchema() const { return _schema; }

#if 0

    /**
     *  @brief Return a strided-array view into the columns of the table.
     *
     *  This will raise LogicErrorException if the table is not consolidated.
     */
    ColumnView getColumnView() const;

#endif

    virtual ~TableBase() {}

protected:

    /// @brief Clone implementation with noncovariant return types.
    virtual PTR(TableBase) _clone() const = 0;

    explicit TableBase(Schema const & schema) : _schema(schema) {}

    TableBase(TableBase const & other) : _schema(other._schema) {}

private:
    
    friend class RecordBase;

    // Called by RecordBase ctor to fill in its _data and _manager members.
    void _initialize(RecordBase & record) const;

    // Tables may be copy-constructable (and are definitely cloneable), but are not assignable.
    void operator=(TableBase const & other) {
        _schema = other._schema;
    }

    Schema _schema;
    mutable ndarray::Manager::Ptr _manager;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_TableBase_h_INCLUDED
