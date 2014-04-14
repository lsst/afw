// -*- lsst-c++ -*-

#include "boost/make_shared.hpp"

#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

// =============== BaseTableImpl and BaseRecordImpl =========================================================

//  These are a private table/record pair -- they're what you actually get when you do TableBase::make(),
//  but we hide them here to avoid a giant nest of friending that would be necessary if they had to make
//  their constructors private or protected.

namespace {

class BaseRecordImpl;

class BaseTableImpl : public BaseTable {
public:

    explicit BaseTableImpl(Schema const & schema) : BaseTable(schema) {}

    BaseTableImpl(BaseTableImpl const & other) : BaseTable(other) {}

private:
    virtual PTR(BaseTable) _clone() const;
    virtual PTR(BaseRecord) _makeRecord();
};

class BaseRecordImpl : public BaseRecord {
public:
    explicit BaseRecordImpl(PTR(BaseTable) const & table) : BaseRecord(table) {}
};

PTR(BaseTable) BaseTableImpl::_clone() const {
    return boost::make_shared<BaseTableImpl>(*this);
}

PTR(BaseRecord) BaseTableImpl::_makeRecord() {
    return boost::make_shared<BaseRecordImpl>(shared_from_this());
}

} // anonymous

// =============== Block ====================================================================================

//  This is a block of memory that doles out record-sized chunks when a table asks for them.
//  It inherits from ndarray::Manager so we can return ndarrays that refer to the memory in the
//  block with correct reference counting (ndarray::Manager is just an empty base class with an
//  internal reference count - it's like a shared_ptr without the pointer and template parameter.
//
//  Records are allocated in Blocks for two reasons:
//    - it allows tables to be either totally contiguous in memory (enabling column views) or
//      not (enabling dynamic addition of records) all in one class.
//    - it saves us from ever having to reallocate all the records associated with a table
//      when we run out of space (that's what a std::vector-like model would require).  This keeps
//      records and/or iterators to them from being invalidated, and it keeps tables from having
//      to track all the records whose data it owns.

namespace {

class Block : public ndarray::Manager {
public:
    typedef boost::intrusive_ptr<Block> Ptr;

    // If the last chunk allocated isn't needed after all (usually because of an exception in a constructor)
    // we reuse it immediately.  If it wasn't the last chunk allocated, it can't be reclaimed until
    // the entire block goes out of scope.
    static void reclaim(std::size_t recordSize, void * data, ndarray::Manager::Ptr const & manager) {
        Ptr block = boost::static_pointer_cast<Block>(manager);
        if (reinterpret_cast<char*>(data) + recordSize == block->_next) {
            block->_next -= recordSize;
        }
    }

    // Ensure we have space for at least the given number of records as a contiguous block.
    // May not actually allocate anything if we already do.
    static void preallocate(
        std::size_t recordSize,
        std::size_t recordCount,
        ndarray::Manager::Ptr & manager
    ) {
        Ptr block = boost::static_pointer_cast<Block>(manager);
        if (!block || static_cast<std::size_t>(block->_end - block->_next) < recordSize * recordCount) {
            block = Ptr(new Block(recordSize, recordCount));
            manager = block;
        }
    }

    static std::size_t getBufferSize(
        std::size_t recordSize,
        ndarray::Manager::Ptr const & manager
    ) {
        Ptr block = boost::static_pointer_cast<Block>(manager);
        return static_cast<std::size_t>(block->_end - block->_next) / recordSize;
    }

    // Get the next chunk from the block, making a new block and installing it into the table
    // if we're all out of space.
    static void * get(std::size_t recordSize, ndarray::Manager::Ptr & manager) {
        Ptr block = boost::static_pointer_cast<Block>(manager);
        if (!block || block->_next == block->_end) {
            block = Ptr(new Block(recordSize, BaseTable::nRecordsPerBlock));
            manager = block;
        }
        void * r = block->_next;
        block->_next += recordSize;
        return r;
    }

    // Block is also keeper of the special number that says what alignment boundaries are needed for
    // schemas.  Before we start using a schema, we need to first ensure it meets that requirement,
    // and pad it if not.
    static void padSchema(Schema & schema) {
        static int const MIN_RECORD_ALIGN = sizeof(AllocType);
        int remainder = schema.getRecordSize() % MIN_RECORD_ALIGN;
        if (remainder) {
            detail::Access::padSchema(schema, MIN_RECORD_ALIGN - remainder);
        }
    }

private:

    struct AllocType {
        double element[2];
    };

    explicit Block(std::size_t recordSize, std::size_t recordCount) :
        _mem(new AllocType[(recordSize * recordCount) / sizeof(AllocType)]),
        _next(reinterpret_cast<char*>(_mem.get())),
        _end(_next + recordSize * recordCount)
    {
        assert((recordSize * recordCount) % sizeof(AllocType) == 0);
        std::fill(_next, _end, 0); // initialize to zero; we'll later initialize floats to NaN.
    }

    boost::scoped_array<AllocType> _mem;
    char * _next;
    char * _end;
};

} // anonymous

// =============== BaseTable implementation (see header for docs) ===========================================

void BaseTable::preallocate(std::size_t n) {
    Block::preallocate(_schema.getRecordSize(), n, _manager);
}

std::size_t BaseTable::getBufferSize() const {
    if (_manager) {
        return Block::getBufferSize(_schema.getRecordSize(), _manager);
    } else {
        return 0;
    }
}

PTR(BaseTable) BaseTable::make(Schema const & schema) {
    return boost::make_shared<BaseTableImpl>(schema);
}

PTR(BaseRecord) BaseTable::copyRecord(BaseRecord const & input) {
    PTR(BaseRecord) output = makeRecord();
    output->assign(input);
    return output;
}

PTR(BaseRecord) BaseTable::copyRecord(BaseRecord const & input, SchemaMapper const & mapper) {
    PTR(BaseRecord) output = makeRecord();
    output->assign(input, mapper);
    return output;
}

PTR(io::FitsWriter) BaseTable::makeFitsWriter(fits::Fits * fitsfile, int flags) const {
    return boost::make_shared<io::FitsWriter>(fitsfile, flags);
}

BaseTable::BaseTable(Schema const & schema) : daf::base::Citizen(typeid(this)), _schema(schema), _version(1) {
    Block::padSchema(_schema);
}

namespace {

// A Schema Functor used to set floating point-fields to NaN.  All others are left 0.
struct RecordInitializer {
    
    template <typename T>
    static void fill(T * element, int size) {} // this matches all non-floating-point-element fields.

    static void fill(float * element, int size) {
        std::fill(element, element + size, std::numeric_limits<float>::quiet_NaN());
    }

    static void fill(double * element, int size) {
        std::fill(element, element + size, std::numeric_limits<double>::quiet_NaN());
    }

    static void fill(Angle * element, int size) {
        fill(reinterpret_cast<double*>(element), size);
    }

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        fill(
            reinterpret_cast<typename Field<T>::Element *>(data + item.key.getOffset()),
            item.key.getElementCount()
        );
    }

    void operator()(SchemaItem<Flag> const & item) const {} // do nothing for Flag fields; already 0

    char * data;
};

} // anonymous

void BaseTable::_initialize(BaseRecord & record) {
    record._data = Block::get(_schema.getRecordSize(), _manager);
    RecordInitializer f = { reinterpret_cast<char*>(record._data) };
    _schema.forEach(f);
    record._manager = _manager; // manager always points to the most recently-used block.
}

void BaseTable::_destroy(BaseRecord & record) {
    assert(record._table.get() == this);
    if (record._manager == _manager) Block::reclaim(_schema.getRecordSize(), record._data, _manager);
}

/*
 *  JFB has no idea whether the default value below is sensible, or even whether
 *  it should be expressed ultimately as an approximate size in bytes rather than a
 *  number of records; the answer probably depends on both the typical size of
 *  records and the typical number of records.
 */
int BaseTable::nRecordsPerBlock = 100;

// =============== BaseCatalog instantiation =================================================================

template class CatalogT<BaseRecord>;
template class CatalogT<BaseRecord const>;

}}} // namespace lsst::afw::table
