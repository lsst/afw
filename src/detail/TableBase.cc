#include <cstring>

#include "boost/noncopyable.hpp"
#include "boost/make_shared.hpp"

#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/detail/TableBase.h"
#include "lsst/afw/table/detail/TreeIteratorBase.h"
#include "lsst/afw/table/detail/IteratorBase.h"
#include "lsst/afw/table/detail/Access.h"


namespace lsst { namespace afw { namespace table { namespace detail {

//----- Block definition and implementation -----------------------------------------------------------------

namespace {

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

    static void padLayout(Layout & layout) {
        static int const MIN_RECORD_ALIGN = sizeof(AllocType);
        Access::padLayout(layout, (MIN_RECORD_ALIGN - layout.getRecordSize() % MIN_RECORD_ALIGN));
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
    int defaultBlockRecordCount;
    RecordData * front;
    RecordData * back;
    void * consolidated;
    Block::Ptr block;
    IdFactory::Ptr idFactory;
    AuxBase::Ptr aux;
    Layout layout;
    RecordSet records;

    void addBlock(int blockRecordCount);

    void assertEqual(boost::shared_ptr<TableImpl> const & other) const {
        if (other.get() != this) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                "Record and/or Iterator is not associated with this table."
            );
        }
    }

    RecordData * addRecord(RecordId id, RecordData * parent, AuxBase::Ptr const & aux);

    void unlink(RecordData * record);

    TableImpl(
        Layout const & layout_, int defaultBlockRecordCount_, 
        IdFactory::Ptr const & idFactory_, AuxBase::Ptr const & aux_
    ) :
        defaultBlockRecordCount(defaultBlockRecordCount_), front(0), back(0), consolidated(0), 
        idFactory(idFactory_), aux(aux_), layout(layout_)
    {
        Block::padLayout(layout);
        if (!idFactory) idFactory = IdFactory::makeSimple();
    }

    ~TableImpl() { records.clear(); }

};

//----- TableImpl implementation ----------------------------------------------------------------------------

void TableImpl::addBlock(int blockRecordCount) {
    Block::Ptr newBlock = Block::allocate(layout.getRecordSize(), blockRecordCount);
    if (block) {
        newBlock->chain.swap(block);
        consolidated = 0;
    } else {
        consolidated = newBlock->getBuffer();
    }
    block.swap(newBlock);
}

RecordData * TableImpl::addRecord(RecordId id, RecordData * parent, AuxBase::Ptr const & aux) {
    RecordSet::insert_commit_data insertData;
    if (!records.insert_check(id, CompareRecordIdLess(), insertData).second) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Record ID '%lld' is not unique.") % id).str()
        );
    }
    if (!block || block->isFull()) {
        addBlock(defaultBlockRecordCount);
    }
    RecordData * p = block->makeNextRecord();
    assert(p != 0);
    p->id = id;
    p->aux = aux;
    if (parent) {
        if (parent->links.child) {
            RecordData * q = parent->links.child;
            while (q->links.next) {
                q = q->links.next;
            }
            q->links.next = p;
            p->links.previous = q;
        } else {
            parent->links.child = p;
        }
        p->links.parent = parent;
    } else {
        if (back == 0) {
            assert(front == 0);
            front = p;
        } else {
            assert(back->links.next == 0);
            back->links.next = p;
            p->links.previous = back;
        }
        back = p;
    }
    records.insert_commit(*p, insertData);
    return p;
}

void TableImpl::unlink(RecordData * record) {
    if (record->links.child) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Children must be erased before parent record."
        );
    }
    if (!record->is_linked()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Record has already been unlinked."
        );
    }
    if (record->links.previous) {
        record->links.previous->links.next = record->links.next;
    } else if (record->links.parent) {
        record->links.parent->links.child = record->links.next;
    }
    if (record->links.next) {
        record->links.next->links.previous = record->links.previous;
    }
    record->links.parent = 0;
    consolidated = 0;
}

//----- TreeIteratorBase implementation ---------------------------------------------------------------------

void TreeIteratorBase::increment() {
    switch (_mode) {
    case DEPTH_FIRST:
        if (_record._data->links.child) {
            _record._data = _record._data->links.child;
        } else if (_record._data->links.next) {
            _record._data = _record._data->links.next;
        } else {
            while (!_record._data->links.next && _record._data->links.parent) {
                _record._data = _record._data->links.parent;
            }
            _record._data = _record._data->links.next;
        }
        break;
    case NO_NESTING:
        _record._data = _record._data->links.next;
        break;
    }
}

//----- IteratorBase implementation ----------------------------------------------------------------------

IteratorBase::~IteratorBase() {}

//----- RecordBase implementation ---------------------------------------------------------------------------

Layout RecordBase::getLayout() const { return _table->layout; }

RecordBase::~RecordBase() {}

TreeIteratorBase RecordBase::_beginChildren(TreeMode mode) const {
    return TreeIteratorBase(_data->links.child, _table, *this, mode);
}

TreeIteratorBase RecordBase::_endChildren(TreeMode mode) const {
    return TreeIteratorBase(
        (mode == NO_NESTING || _data->links.child == 0) ? 0 : _data->links.next,
        _table, *this, mode
    );
}

RecordBase RecordBase::_addChild(RecordId id, AuxBase::Ptr const & aux) const {
    assertBit(CAN_ADD_RECORD);
    RecordData * p = _table->addRecord(id, _data, aux);
    _table->idFactory->notify(id);
    return RecordBase(p, _table, *this);
}

RecordBase RecordBase::_addChild(AuxBase::Ptr const & aux) const {
    return _addChild((*_table->idFactory)(), aux);
}

void RecordBase::unlink() const {
    assertBit(CAN_UNLINK_RECORD);
    _table->unlink(_data);
    _table->records.erase(_table->records.s_iterator_to(*_data));
}

//----- TableBase implementation --------------------------------------------------------------------------

Layout TableBase::getLayout() const { return _impl->layout; }

bool TableBase::isConsolidated() const {
    return _impl->consolidated;
}

void TableBase::consolidate(int extraCapacity) {
    boost::shared_ptr<TableImpl> newImpl =
        boost::make_shared<TableImpl>(
            _impl->layout,
            _impl->defaultBlockRecordCount,
            _impl->idFactory->clone(),
            _impl->aux
        );
    newImpl->addBlock(_impl->records.size() + extraCapacity);
    int const dataOffset = sizeof(RecordData);
    int const dataSize = _impl->layout.getRecordSize() - dataOffset;
    for (RecordSet::const_iterator i = _impl->records.begin(); i != _impl->records.end(); ++i) {
        RecordData * newRecord = newImpl->block->makeNextRecord();
        newRecord->id = i->id;
        newRecord->aux = i->aux;
        newRecord->parentId = (i->links.parent) ? i->links.parent->id : 0;
        std::memcpy(newRecord + dataOffset, &(*i) + dataOffset, dataSize);
        newImpl->records.insert(newImpl->records.end(), *newRecord);
    }
    setupPointers(newImpl->records, newImpl->back);
    newImpl->front = &(*_impl->records.begin());
    _impl.swap(newImpl);
}

int TableBase::getRecordCount() const {
    return _impl->records.size();
}

IteratorBase TableBase::_unlink(IteratorBase const & iter) const {
    assertBit(CAN_UNLINK_RECORD);
    _impl->assertEqual(iter._table);
    _impl->unlink(iter->_data);
    return IteratorBase(_impl->records.erase(iter.base()), _impl, iter);
}

TreeIteratorBase TableBase::_unlink(TreeIteratorBase const & iter) const {
    assertBit(CAN_UNLINK_RECORD);
    _impl->assertEqual(iter->_table);
    TreeIteratorBase result(iter);
    ++result;
    _impl->unlink(iter->_data);
    _impl->records.erase(_impl->records.iterator_to(*iter->_data));
    return result;
}

TreeIteratorBase TableBase::_beginTree(TreeMode mode) const {
    return TreeIteratorBase(_impl->front, _impl, *this, mode);
}

TreeIteratorBase TableBase::_endTree(TreeMode mode) const {
    return TreeIteratorBase(0, _impl, *this, mode);
}

IteratorBase TableBase::_begin() const {
    return IteratorBase(_impl->records.begin(), _impl, *this);
}

IteratorBase TableBase::_end() const {
    return IteratorBase(_impl->records.end(), _impl, *this);
}

RecordBase TableBase::_get(RecordId id) const {
    RecordSet::iterator j = _impl->records.find(id, detail::CompareRecordIdLess());
    if (j == _impl->records.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            (boost::format("Record with id '%lld' not found.") % id).str()
        );
    }
    return RecordBase(&(*j), _impl, *this);
}

IteratorBase TableBase::_find(RecordId id) const {
    RecordSet::iterator j = _impl->records.find(id, detail::CompareRecordIdLess());
    return IteratorBase(j, _impl, *this);
}

RecordBase TableBase::_addRecord(AuxBase::Ptr const & aux) const {
    assertBit(CAN_ADD_RECORD);
    return _addRecord((*_impl->idFactory)(), aux);
}

RecordBase TableBase::_addRecord(RecordId id, AuxBase::Ptr const & aux) const {
    assertBit(CAN_ADD_RECORD);
    RecordData * p = _impl->addRecord(id, 0, aux);
    _impl->idFactory->notify(id);
    return RecordBase(p, _impl, *this);
}

TableBase::TableBase(
    Layout const & layout,
    int defaultBlockRecordCount,
    int capacity,
    IdFactory::Ptr const & idFactory,
    AuxBase::Ptr const & aux,
    ModificationFlags const & flags 
) : ModificationFlags(flags),
    _impl(boost::make_shared<TableImpl>(layout, defaultBlockRecordCount, idFactory, aux))
{
    if (capacity > 0) _impl->addBlock(capacity);
}

}}}} // namespace lsst::afw::table::detail
