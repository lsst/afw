// -*- lsst-c++ -*-

#include <cstring>

#include "boost/noncopyable.hpp"
#include "boost/make_shared.hpp"

#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/TableBase.h"
#include "lsst/afw/table/IteratorBase.h"
#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table { namespace detail {

namespace {

//----- Copy functor for copying with mapper ----------------------------------------------------------------

void copyRecord(RecordData const * inputRecord, RecordData * outputRecord, Schema const & schema) {
    std::memcpy(
        reinterpret_cast<char *>(outputRecord) + sizeof(RecordData),
        reinterpret_cast<char const*>(inputRecord) + sizeof(RecordData),
        schema.getRecordSize() - sizeof(RecordData)
    );  
}

struct CopyValue {

    template <typename U>
    void operator()(Key<U> const & inputKey, Key<U> const & outputKey) const {
        Access::copyValue(
            inputKey, _inputRecord,
            outputKey, _outputRecord
        );
    }

    CopyValue(RecordData const * inputRecord, RecordData * outputRecord) :
        _inputRecord(inputRecord), _outputRecord(outputRecord)
    {}

private:
    RecordData const * _inputRecord;
    RecordData * _outputRecord;
};

//----- Block definition and implementation -----------------------------------------------------------------

class Block : public ndarray::Manager {
public:

    typedef boost::intrusive_ptr<Block> Ptr;

    RecordData * makeNextRecord() {
        if (isFull()) return 0;
        RecordData * r = new (_nextLocation) RecordData();
        _nextLocation += _recordSize;
        return r;
    }

    bool isFull() const { return _nextLocation == _end; }

    void * getBuffer() const {
        return _buf.get();
    }

    static void padSchema(Schema & schema) {
        static int const MIN_RECORD_ALIGN = sizeof(AllocType);
        Access::padSchema(schema, (MIN_RECORD_ALIGN - schema.getRecordSize() % MIN_RECORD_ALIGN));
    }

    static Ptr allocate(int recordSize, int recordCount) {
        boost::intrusive_ptr<Block> r(new Block(recordSize, recordCount));
        return r;
    }

    virtual ~Block() {
        char * iter = reinterpret_cast<char*>(_buf.get());
        while (iter != _nextLocation) {
            reinterpret_cast<RecordData*>(iter)->~RecordData();
            iter += _recordSize;
        }
    }

    Ptr chain;

private:

    struct AllocType {
        double element[2];
    };
    
    explicit Block(int recordSize, int recordCount) :
        _recordSize(recordSize),
        _buf(new AllocType[(recordSize * recordCount) / sizeof(AllocType)]),
        _nextLocation(reinterpret_cast<char*>(_buf.get())),
        _end(_nextLocation + recordSize * recordCount)
    {
        assert((recordSize * recordCount) % sizeof(AllocType) == 0);
    }

    int _recordSize;
    boost::scoped_array<AllocType> _buf;
    char * _nextLocation;
    char * _end;
};

} // anonymous

//----- TableImpl definition --------------------------------------------------------------------------------

struct TableImpl : private boost::noncopyable {
    void * consolidated;
    Block::Ptr block;
    PTR(IdFactory) idFactory;
    PTR(AuxBase) aux;
    Schema schema;
    RecordSet records;

    void addBlock(int blockRecordCount);

    void assertEqual(PTR(TableImpl) const & other) const {
        if (other.get() != this) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                "Record and/or Iterator is not associated with this table."
            );
        }
    }

    RecordData * addRecord(
        RecordSet::iterator const & hint,
        RecordId id,
        PTR(AuxBase) const & aux,
        bool returnExisting
    );

    void unlink(RecordData * record);

    TableImpl(Schema const & schema_, PTR(IdFactory) const & idFactory_, PTR(AuxBase) const & aux_) :
        consolidated(0), idFactory(idFactory_), aux(aux_), schema(schema_)
    {
        Block::padSchema(schema);
        if (!idFactory) idFactory = IdFactory::makeSimple();
    }

    ~TableImpl() { records.clear(); }

};

//----- TableImpl implementation ----------------------------------------------------------------------------

void TableImpl::addBlock(int blockRecordCount) {
    Block::Ptr newBlock = Block::allocate(schema.getRecordSize(), blockRecordCount);
    if (block) {
        newBlock->chain.swap(block);
        consolidated = 0;
    } else {
        consolidated = newBlock->getBuffer();
    }
    block.swap(newBlock);
}

RecordData * TableImpl::addRecord(
    RecordSet::iterator const & hint,
    RecordId id,
    PTR(AuxBase) const & aux,
    bool returnExisting
) {
    RecordSet::insert_commit_data insertData;
    std::pair<RecordSet::iterator,bool> i = records.insert_check(hint, id, CompareRecordIdLess(), insertData);
    if (!i.second) {
        if (returnExisting) return &*i.first;
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Record ID '%lld' is not unique.") % id).str()
        );
    }
    if (!block || block->isFull()) {
        addBlock(TableBase::nRecordsPerBlock);
    }
    RecordData * p = block->makeNextRecord();
    assert(p != 0);
    p->id = id;
    p->aux = aux;
    records.insert_commit(*p, insertData);
    return p;
}

void TableImpl::unlink(RecordData * record) {
    if (!record->is_linked()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record has already been unlinked."
        );
    }
    detail::Access::getParentId(schema, *record) = 0;
    consolidated = 0;
}

} // namespace detail

//----- IteratorBase implementation ----------------------------------------------------------------------

IteratorBase::~IteratorBase() {}

//----- RecordBase implementation ---------------------------------------------------------------------------

Schema RecordBase::getSchema() const { return _table->schema; }

RecordBase::~RecordBase() {}

RecordId RecordBase::getParentId() const {
    if (!_table->schema.hasTree()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record's schema has no parent ID."
        );
    }
    return detail::Access::getParentId(_table->schema, *_data);
}

void RecordBase::setParentId(RecordId id) const {
    if (!_table->schema.hasTree()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record's schema has no parent ID."
        );
    }
    detail::Access::getParentId(_table->schema, *_data) = id;
}

bool RecordBase::hasParent() const {
    if (!_table->schema.hasTree()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record's schema has no parent ID."
        );
    }
    return detail::Access::getParentId(_table->schema, *_data) != 0;
}

bool RecordBase::hasChildren() const {
    if (!_table->schema.hasTree()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record's schema has no parent ID."
        );
    }
    for (detail::RecordSet::iterator i = _table->records.begin(); i != _table->records.end(); ++i) {
        if (detail::Access::getParentId(_table->schema, *i) == _data->id) {
            return true;
        }
    }
    return false;
}

PTR(AuxBase) RecordBase::getTableAux() const { return _table->aux; }

RecordBase RecordBase::_getParent() const {
    if (!_table->schema.hasTree()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record's schema has no parent ID."
        );
    }
    RecordId parentId = detail::Access::getParentId(_table->schema, *_data);
    if (!parentId) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            "Record has no parent."
        );
    }
    detail::RecordSet::iterator j = _table->records.find(parentId, detail::CompareRecordIdLess());
    if (j == _table->records.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            (boost::format("Parent record with id '%lld' not found.") % parentId).str()
        );
    }
    return RecordBase(&(*j), _table, *this);
}

ChildIteratorBase RecordBase::_beginChildren() const {
    if (!_table->schema.hasTree()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record's schema has no parent ID."
        );
    }
    return ChildIteratorBase(
        detail::ChildFilterPredicate(_data->id),
        IteratorBase(_table->records.s_iterator_to(*_data), _table, *this),
        IteratorBase(_table->records.end(), _table, *this)
    );
}

void RecordBase::_copyFrom(RecordBase const & other) const {
    assertBit(CAN_SET_FIELD);
    if (other.getSchema() != _table->schema) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Input record's schema does not match output record's Schema."
        );
    }
    _data->aux = other._data->aux;
    detail::copyRecord(other._data, _data, _table->schema);
}

void RecordBase::_copyFrom(RecordBase const & other, SchemaMapper const & mapper) const {
    assertBit(CAN_SET_FIELD);
    if (other.getSchema() != mapper.getInputSchema()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Input record's schema does not match mapper's input Schema."
        );
    }
    if (_table->schema != mapper.getOutputSchema()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Output record's schema does not match mapper's output Schema."
        );
    }
    if (_table->schema.hasTree() && other.getSchema().hasTree()) {
        detail::Access::getParentId(_table->schema, *_data)
            = detail::Access::getParentId(_table->schema, *other._data);
    }
    _data->aux = other._data->aux;
    mapper.forEach(detail::CopyValue(other._data, _data));
}

ChildIteratorBase RecordBase::_endChildren() const {
    if (!_table->schema.hasTree()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record's schema has no parent ID."
        );
    }
    return ChildIteratorBase(
        detail::ChildFilterPredicate(_data->id),
        IteratorBase(_table->records.end(), _table, *this),
        IteratorBase(_table->records.end(), _table, *this)
    );
}

void RecordBase::unlink() const {
    assertBit(CAN_UNLINK_RECORD);
    _table->unlink(_data);
    _table->records.erase(_table->records.s_iterator_to(*_data));
}

void RecordBase::operator=(RecordBase const & other) {
    if (_table && _table->schema != other._table->schema) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Invalid record assignment: schemas are not equal."
        );
    }
    _data = other._data;
    _table = other._table;
}

//----- TableBase implementation --------------------------------------------------------------------------


/*
 *  The author has no idea whether the default value below is sensible, or even whether
 *  it should be expressed ultimately as an approximate size in bytes rather than a
 *  number of records; the answer probably depends on both the typical size of
 *  records and the typical number of records.
 */
int TableBase::nRecordsPerBlock = 100;

TableBase::~TableBase() {}

Schema TableBase::getSchema() const { return _impl->schema; }

bool TableBase::isConsolidated() const {
    return _impl->consolidated;
}

void TableBase::consolidate(int extraCapacity) {
    PTR(detail::TableImpl) newImpl =
        boost::make_shared<detail::TableImpl>(
            _impl->schema,
            _impl->idFactory->clone(),
            _impl->aux
        );
    newImpl->addBlock(_impl->records.size() + extraCapacity);
    for (detail::RecordSet::iterator i = _impl->records.begin(); i != _impl->records.end(); ++i) {
        detail::RecordData * newRecord = newImpl->block->makeNextRecord();
        newRecord->id = i->id;
        newRecord->aux = i->aux;
        detail::copyRecord(&*i, newRecord, _impl->schema);
        newImpl->records.insert(newImpl->records.end(), *newRecord);
    }
    _impl.swap(newImpl);
}

ColumnView TableBase::getColumnView() const {
    if (!isConsolidated()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "getColumnView() can only be called on a consolidated table"
        );
    }
    return ColumnView(_impl->schema, _impl->records.size(), _impl->consolidated, _impl->block);
}

int TableBase::getRecordCount() const {
    return _impl->records.size();
}

IteratorBase TableBase::unlink(IteratorBase const & iter) const {
    assertBit(CAN_UNLINK_RECORD);
    _impl->assertEqual(iter._table);
    _impl->unlink(iter->_data);
    return IteratorBase(_impl->records.erase(iter.base()), _impl, iter);
}

IteratorBase TableBase::begin() const {
    return IteratorBase(_impl->records.begin(), _impl, *this);
}

IteratorBase TableBase::end() const {
    return IteratorBase(_impl->records.end(), _impl, *this);
}

RecordBase TableBase::operator[](RecordId id) const {
    detail::RecordSet::iterator j = _impl->records.find(id, detail::CompareRecordIdLess());
    if (j == _impl->records.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            (boost::format("Record with id '%lld' not found.") % id).str()
        );
    }
    return RecordBase(&(*j), _impl, *this);
}

IteratorBase TableBase::find(RecordId id) const {
    detail::RecordSet::iterator j = _impl->records.find(id, detail::CompareRecordIdLess());
    return IteratorBase(j, _impl, *this);
}

PTR(AuxBase) & TableBase::getAux() const { return _impl->aux; }

RecordBase TableBase::_addRecord(PTR(AuxBase) const & aux) const {
    assertBit(CAN_ADD_RECORD);
    return _addRecord((*_impl->idFactory)(), aux);
}

RecordBase TableBase::_addRecord(RecordId id, PTR(AuxBase) const & aux) const {
    assertBit(CAN_ADD_RECORD);
    detail::RecordData * p = _impl->addRecord(_impl->records.end(), id, aux, false);
    _impl->idFactory->notify(id);
    return RecordBase(p, _impl, *this);
}

IteratorBase TableBase::_insert(IteratorBase const & hint, RecordBase const & record) const {
    assertBit(CAN_ADD_RECORD);
    assertBit(CAN_SET_FIELD);
    detail::RecordData * p = _impl->addRecord(hint.base(), record._data->id, record._data->aux, true);
    IteratorBase result(_impl->records.s_iterator_to(*p), _impl, *this);
    result->_copyFrom(record);
    return result;
}

IteratorBase TableBase::_insert(
    IteratorBase const & hint, RecordBase const & record, SchemaMapper const & mapper
) const {
    assertBit(CAN_ADD_RECORD);
    assertBit(CAN_SET_FIELD);
    detail::RecordData * p = _impl->addRecord(hint.base(), record._data->id, record._data->aux, true);
    IteratorBase result(_impl->records.s_iterator_to(*p), _impl, *this);
    result->_copyFrom(record, mapper);
    return result;
}

TableBase::TableBase(
    Schema const & schema,
    int capacity,
    PTR(IdFactory) const & idFactory,
    PTR(AuxBase) const & aux,
    ModificationFlags const & flags 
) : ModificationFlags(flags),
    _impl(boost::make_shared<detail::TableImpl>(schema, idFactory, aux))
{
    if (capacity > 0) _impl->addBlock(capacity);
}

}}} // namespace lsst::afw::table
