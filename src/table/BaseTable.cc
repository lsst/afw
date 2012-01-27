// -*- lsst-c++ -*-

#include "boost/make_shared.hpp"

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/Vector.h"
#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

namespace {

class SimpleRecord;

class SimpleTable : public BaseTable {
public:

    explicit SimpleTable(Schema const & schema) : BaseTable(schema) {}

    SimpleTable(SimpleTable const & other) : BaseTable(other) {}

private:
    virtual PTR(BaseTable) _clone() const;
    virtual PTR(BaseRecord) _makeRecord();
};

class SimpleRecord : public BaseRecord {
public:
    explicit SimpleRecord(PTR(BaseTable) const & table) : BaseRecord(table) {}
};

PTR(BaseTable) SimpleTable::_clone() const {
    return boost::make_shared<SimpleTable>(*this);
}

PTR(BaseRecord) SimpleTable::_makeRecord() {
    return boost::make_shared<SimpleRecord>(shared_from_this());
}

class Block : public ndarray::Manager {
public:
    typedef boost::intrusive_ptr<Block> Ptr;

    static void reclaim(std::size_t recordSize, void * data, ndarray::Manager::Ptr const & manager) {
        Ptr block = boost::static_pointer_cast<Block>(manager);
        if (reinterpret_cast<char*>(data) + recordSize == block->_next) {
            block->_next -= recordSize;
        }
    }

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
    }

    boost::scoped_array<AllocType> _mem;
    char * _next;
    char * _end;
};

} // anonymous

void BaseTable::preallocate(std::size_t n) {
    Block::preallocate(_schema.getRecordSize(), n, _manager);
}

PTR(BaseTable) BaseTable::make(Schema const & schema) {
    return boost::make_shared<SimpleTable>(schema);
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

PTR(io::FitsWriter) BaseTable::makeFitsWriter(io::FitsWriter::Fits * fits) const {
    return boost::make_shared<io::FitsWriter>(fits);
}

BaseTable::BaseTable(Schema const & schema) : _schema(schema) {
    Block::padSchema(_schema);
}

void BaseTable::_initialize(BaseRecord & record) {
    record._data = Block::get(_schema.getRecordSize(), _manager);
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

template class Vector<BaseRecord>;
template class Vector<BaseRecord const>;

}}} // namespace lsst::afw::table
