// -*- lsst-c++ -*-
#ifndef AFW_TABLE_TableBase_h_INCLUDED
#define AFW_TABLE_TableBase_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/TreeIteratorBase.h"
#include "lsst/afw/table/IteratorBase.h"
#include "lsst/afw/table/IdFactory.h"

namespace lsst {

namespace daf { namespace base {

class PropertySet;

}} // namespace daf::base

namespace afw { namespace table {

class LayoutMapper;

/**
 *  @brief Base class containing most of the implementation for tables.
 *
 *  Most of the implementation of derived table classes is provided here
 *  in the form of protected member functions that will need to be wrapped
 *  into public member functions by derived classes.
 *
 *  Data is shared between records and tables, but the assertion-based
 *  modification flags are not shared.
 */
class TableBase : protected ModificationFlags {
public:

    /// @brief Return the layout for the table's fields.  
    Layout getLayout() const;

    /// @brief Return true if all records are allocated in a single contiguous blocks.
    bool isConsolidated() const;

    /**
     *  @brief Consolidate the table in-place into a single contiguous block.
     *
     *  This does not invalidate any existing records or iterators, but existing
     *  records and iterators will no longer be associated with this table.
     *
     *  This will also reallocate the table even if the table is already consolidated.
     *  
     *  @param[in] extraCapacity  Number of additional records to allocate space for
     *                            as part of the same block.  Adding N additional records
     *                            where N is <= extraCapacity will not cause the table
     *                            to become unconsolidated.
     */
    void consolidate(int extraCapacity=0);

    /**
     *  @brief Return a strided-array view into the columns of the table.
     *
     *  This will raise LogicErrorException if the table is not consolidated.
     */
    ColumnView getColumnView() const;

    /// @brief Return the number of records in the table.
    int getRecordCount() const;

    /// @brief Disable modifications of the sort defined by the given bit.
    void disable(ModificationFlags::Bit n) { unsetBit(n); }

    /// @brief Disable all modifications.
    void makeReadOnly() { unsetAll(); }

    /// Destructor is explicit because class holds a shared_ptr to an incomplete class.
    ~TableBase();

protected:

    /**
     *  @brief Standard constructor for TableBase.
     *
     *  @param[in] layout            Layout that defines the fields, offsets, and record size for the table.
     *  @param[in] nRecordsPerBlock  Number of records to allocate space for in each block.  This is almost
     *                               entirely a performance-only consideration, but it does affect whether
     *                               a table will be remain consolidated after adding records.
     *  @param[in] capacity          Number of records to pre-allocate space for in the first block.  This
     *                               overrides nRecordsPerBlock for the first block and the first block only.
     *  @param[in] idFactory         Factory class to generate record IDs when they are not explicitly given.
     *                               If empty, defaults to a simple counter that starts at 1.
     *  @param[in] aux               A pointer containing extra arbitrary data for the table.
     *  @param[in] flags             Bitflags for assertion-based modification protection (see the
     *                               ModificationFlags class for more information).
     */
    TableBase(
        Layout const & layout,
        int nRecordsPerBlock,
        int capacity,
        IdFactory::Ptr const & idFactory = IdFactory::Ptr(),
        AuxBase::Ptr const & aux = AuxBase::Ptr(),
        ModificationFlags const & flags = ModificationFlags::all()
    );

    /**
     *  @brief Shared copy constructor.
     *
     *  All aspects of the table except the modification flags are shared between the two tables.
     *  The modification flags will be copied as well, but can then be changed separately.
     */
    TableBase(TableBase const & other) : ModificationFlags(other), _impl(other._impl) {}

    /**
     *  @brief Write the table to a FITS binary table.
     *
     *  Signature is based on the image FITS interface right now; it may be changed 
     *  as I learn what's possible in the implementation.
     */
    void _writeFits(
        std::string const & name,
        CONST_PTR(daf::base::PropertySet) const & metadata = CONST_PTR(daf::base::PropertySet)(),
        std::string const & mode = "w"
    ) const;

    /**
     *  @brief Write the table to a FITS binary table.
     *
     *  This version will use a LayoutMapper to write a subset of the fields to FITS,
     *  and also allows the saved fields to have different names and descriptions.
     *
     *  Signature is based on the image FITS interface right now; it may be changed 
     *  as I learn what's possible in the implementation.
     */
    void _writeFits(
        std::string const & name,
        LayoutMapper const & mapper,
        CONST_PTR(daf::base::PropertySet) const & metadata = CONST_PTR(daf::base::PropertySet)(),
        std::string const & mode = "w"
    ) const;

    //@{
    /**
     *  @brief Remove the record pointed at by the given iterator.
     *
     *  After unlinking, the iterator can still be dereferenced and the record will remain valid,
     *  but the result of incrementing the iterator is undefined.
     */
    IteratorBase _unlink(IteratorBase const & iter) const;
    TreeIteratorBase _unlink(TreeIteratorBase const & iter) const;
    //@}

    //@{
    /// @brief Return begin and end iterators that go through the table in a parent/child-aware way.
    TreeIteratorBase _beginTree(TreeMode mode) const;
    TreeIteratorBase _endTree(TreeMode mode) const;
    //@}

    //@{
    /// @brief Return begin and end iterators that go through the table in ID order.
    IteratorBase _begin() const;
    IteratorBase _end() const;
    //@}

    /// @brief Return the record with the given ID or throw NotFoundException.
    RecordBase _get(RecordId id) const;

    /// @brief Return an iterator to the record with the given ID or throw NotFoundException.
    IteratorBase _find(RecordId id) const;

    /// @brief Create and add a new record with an ID generated by the table's IdFactory.
    RecordBase _addRecord(AuxBase::Ptr const & aux = AuxBase::Ptr()) const;

    /// @brief Create and add a new record with an explicit RecordId.
    RecordBase _addRecord(RecordId id, AuxBase::Ptr const & aux = AuxBase::Ptr()) const;

    /// @brief Return the table's auxiliary data.
    AuxBase::Ptr getAux() const;

private:
    boost::shared_ptr<detail::TableImpl> _impl;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_TableBase_h_INCLUDED
