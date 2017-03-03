// -*- lsst-c++ -*-
#ifndef AFW_TABLE_BaseTable_h_INCLUDED
#define AFW_TABLE_BaseTable_h_INCLUDED
#include <memory>

#include "lsst/base.h"
#include "lsst/daf/base/Citizen.h"
#include "ndarray/Manager.h"
#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw {

namespace fits {

class Fits;

} // namespace fits

namespace table {

/**
 *  @brief Base class for all tables.
 *
 *  Tables have two largely distinct purposes:
 *   - They serve as factories for records, allocating their field data in blocks.
 *   - They carry additional information (such as the schema) that should be shared by multiple records.
 *
 *  It's mostly a matter of convenience that we use the same class to serve both needs.
 *
 *  Tables do not actually maintain a list of all the records they have allocated - but those records
 *  hold a pointer back to the table.  This allows the work of holding and iterating over records to
 *  be delegated to templated container classes (such as CatalogT) while allowing tables to be polymorphic,
 *  non-template classes.  In some sense, then, it may make more sense to think of a table as a combination
 *  factory and "container header".
 *
 *  Tables are always created in shared_ptrs (a requirement of enable_shared_from_this).  BaseTable provides
 *  a make static member function to create a new table, and most derived table classes should do the same.
 *
 *  Each table class should be associated with a particular record class (1-to-1).  Each table instance may
 *  be associated with many record instances.
 */
class BaseTable
#ifndef SWIG // swig complains about these not being %shared_ptrs, but it doesn't need to know about them
: public std::enable_shared_from_this<BaseTable>,
  public daf::base::Citizen
#endif
{
public:

    /// The associated record class.
    typedef BaseRecord Record;

    /// The associated ColumnView class.
    typedef BaseColumnView ColumnView;

    /// Template of CatalogT used to hold records of the associated type.
    typedef CatalogT<Record> Catalog;

    /// Template of CatalogT used to hold const records of the associated type.
    typedef CatalogT<Record const> ConstCatalog;

    /// @brief Number of records in each memory block.
    static int nRecordsPerBlock;

    /// @brief Return the flexible metadata associated with the table.  May be null.
    PTR(daf::base::PropertyList) getMetadata() const { return _metadata; }

    /// @brief Set the flexible metadata associated with the table.  May be null.
    void setMetadata(PTR(daf::base::PropertyList) const & metadata) { _metadata = metadata; }

    /// @brief Return the metadata and set the internal metadata to a null pointer.
    PTR(daf::base::PropertyList) popMetadata() {
        PTR(daf::base::PropertyList) tmp;
        _metadata.swap(tmp);
        return tmp;
    }

    /**
     *  @brief Return a polymorphic deep copy of the table.
     *
     *  Derived classes should reimplement by static-casting the output of _clone to a
     *  pointer-to-derived to simulate covariant return types.
     *
     *  Cloning a table does not clone its associated records; the new table produced by clone()
     *  does not have any associated records.
     */
    PTR(BaseTable) clone() const { return _clone(); }

    /**
     *  @brief Default-construct an associated record.
     *
     *  Derived classes should reimplement by static-casting the output of _makeRecord to the
     *  appropriate BaseRecord subclass to simulate covariant return types.
     */
    PTR(BaseRecord) makeRecord() { return _makeRecord(); }

    /**
     *  @brief Deep-copy a record, requiring that it have the same schema as this table.
     *
     *  Regardless of the type or associated table of the input record, the type of the output record
     *  will be the type associated with this table and the record instance will be associated with
     *  this table.
     *
     *  Allowing derived-class records to be constructed from base-class records could be considered
     *  a form of type-slicing, but because we already demand that all records be constructable from
     *  nothing but a table, this isn't anything new.
     *
     *  Derived classes should reimplement by static-casting the output of BaseTable::copyRecord to the
     *  appropriate BaseRecord subclass.
     *
     *  This is implemented using makeRecord and calling record.assign on the results; override those
     *  to change the behavior.
     */
    PTR(BaseRecord) copyRecord(BaseRecord const & input);

    /**
     *  @brief Deep-copy a record, using a mapper to relate two schemas.
     *
     *  @copydetails BaseTable::copyRecord(BaseRecord const &)
     */
    PTR(BaseRecord) copyRecord(BaseRecord const & input, SchemaMapper const & mapper);

    /// @brief Return the table's schema.
    Schema getSchema() const { return _schema; }

    /**
     *  @brief Allocate contiguous space for new records in advance.
     *
     *  If a contiguous memory block for at least n additional records has already been allocated,
     *  this is a no-op.  If not, a new block will be allocated, and any remaining space on the old
     *  block will go to waste; this ensures the new records will be allocated contiguously.  Note
     *  that "wasted" memory is not leaked; it will be deallocated along with any records
     *  created from that block when those records go out of scope.
     *
     *  Note that unlike std::vector::reserve, this does not factor in existing records in any way;
     *  nRecords refers to a number of new records to reserve space for.
     */
    void preallocate(std::size_t nRecords);

    /**
     *  @brief Return the number of additional records space has been already been allocated for.
     *
     *  Unlike std::vector::capacity, this does not factor in existing records in any way.
     */
    std::size_t getBufferSize() const;

    /**
     *  @brief Construct a new table.
     *
     *  Because BaseTable is an abstract class, this actually returns a hidden trivial subclass
     *  (which is associated with a hidden trivial subclass of BaseRecord).
     *
     *  Hiding concrete table and record classes in anonymous namespaces is not required, but it
     *  makes it easier to ensure instances are always created within shared_ptrs,
     *  and it eliminates some multilateral friending that would otherwise be necessary.
     *  In some cases it may also serve as a form of pimpl, keeping class implementation details
     *  out of header files.
     */
    static PTR(BaseTable) make(Schema const & schema);

    virtual ~BaseTable();

protected:

    /// @brief Convenience function for static-casting shared_from_this for use by derived classes.
    template <typename Derived>
    PTR(Derived) getSelf() {
        return std::static_pointer_cast<Derived>(shared_from_this());
    }

    /// @brief Convenience function for static-casting shared_from_this for use by derived classes.
    template <typename Derived>
    CONST_PTR(Derived) getSelf() const {
        return std::static_pointer_cast<Derived const>(shared_from_this());
    }

    virtual void handleAliasChange(std::string const & alias) {}

    /// Clone implementation with noncovariant return types.
    virtual std::shared_ptr<BaseTable> _clone() const;

    /// Default-construct an associated record (protected implementation).
    virtual std::shared_ptr<BaseRecord> _makeRecord();

    /// @brief Construct from a schema.
    explicit BaseTable(Schema const & schema);

    /// @brief Copy construct.
    BaseTable(BaseTable const & other) :
        daf::base::Citizen(other), _schema(other._schema),
        _metadata(other._metadata)
    {
        if (_metadata)
            _metadata = std::static_pointer_cast<daf::base::PropertyList>(_metadata->deepCopy());
    }

private:

    friend class BaseRecord;
    friend class io::FitsWriter;
    friend class AliasMap;

    // Called by BaseRecord ctor to fill in its _data, _table, and _manager members.
    void _initialize(BaseRecord & record);

    /*
     *  Called by BaseRecord dtor to notify the table when it is about to be destroyed.
     *
     *  This could allow the table to reclaim that space, but that requires more bookkeeping than
     *  it's presently worth unless this was the most recently allocated record.
     *
     *  The motivation for attempting to reclaim even some memory is not because we're paranoid
     *  about using every last bit of allocated memory efficiently - it's so we can keep
     *  records contiguous as much as possible to allow ColumnView to be used.
     */
    void _destroy(BaseRecord & record);

    // Tables are not assignable to prevent type slicing; this is intentionally not implemented,
    // so we get linker errors if we do try to use the assignment operator.
    void operator=(BaseTable const & other);

    // Return a writer object that knows how to save in FITS format.  See also FitsWriter.
    virtual PTR(io::FitsWriter) makeFitsWriter(fits::Fits * fitsfile, int flags) const;

    // All these are definitely private, not protected - we don't want derived classes mucking with them.
    Schema _schema;                 // schema that defines the table's fields
    ndarray::Manager::Ptr _manager; // current memory block to use for new records
    PTR(daf::base::PropertyList) _metadata; // flexible metadata; may be null
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_BaseTable_h_INCLUDED
