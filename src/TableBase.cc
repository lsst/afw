// -*- lsst-c++ -*-

#include <cstring>

#include "boost/noncopyable.hpp"
#include "boost/make_shared.hpp"

#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/TableBase.h"
#include "lsst/afw/table/TreeIteratorBase.h"
#include "lsst/afw/table/IteratorBase.h"
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
    LinkMode linkMode;
    int nRecordsPerBlock;
    RecordData * front;
    RecordData * back;
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

    RecordData * addRecord(RecordId id, RecordData * parent, PTR(AuxBase) const & aux);

    void setLinkMode(LinkMode newMode);

    void unlink(RecordData * record);

    TableImpl(
        Schema const & schema_, int nRecordsPerBlock_, 
        PTR(IdFactory) const & idFactory_, PTR(AuxBase) const & aux_
    ) :
        linkMode(POINTERS), nRecordsPerBlock(nRecordsPerBlock_), front(0), back(0), consolidated(0), 
        idFactory(idFactory_), aux(aux_), schema(schema_)
    {
        Block::padSchema(schema);
        if (!idFactory) idFactory = IdFactory::makeSimple();
    }

    ~TableImpl() { records.clear(); }

};

//----- TableImpl implementation ----------------------------------------------------------------------------

void TableImpl::setLinkMode(LinkMode newMode) {
    if (newMode == linkMode) return;
    if (newMode == PARENT_ID) {
        for (RecordSet::iterator i = records.begin(); i != records.end(); ++i) {
            if (i->links.parent) {
                RecordId parentId = i->links.parent->id;
                i->parentId = parentId;
            }
        }
        front = 0;
        back = 0;
    } else {
        // These pointers optimize the case where children of a common parent have contiguous IDs.
        RecordData * parent = 0;
        RecordData * sibling = 0;
        if (!records.empty()) front = &(*records.begin());
        for (RecordSet::iterator i = records.begin(); i != records.end(); ++i) {
            if (i->parentId) {
                if (!parent || parent->id != i->parentId) {
                    if (i->parentId >= i->id) {
                        throw LSST_EXCEPT(
                            lsst::pex::exceptions::LogicErrorException,
                            (boost::format(
                                "All child records must have IDs strictly greater than their parents; "
                                "%lld >= %lld"
                            ) % i->parentId % i->id).str()
                        );
                    }
                    RecordSet::iterator p = records.find(i->parentId, CompareRecordIdLess());                
                    if (p == records.end()) {
                        throw LSST_EXCEPT(
                            lsst::pex::exceptions::NotFoundException,
                            (boost::format(
                                "Parent record %lld not found for child %lld."
                            ) % i->parentId % i->id).str()
                        );
                    }
                    parent = &(*p);
                    sibling = 0;
                }
                i->links.initialize();
                if (!parent->links.child) {
                    parent->links.child = &(*i);
                } else {
                    if (!sibling) {
                        sibling = parent->links.child;
                        while (sibling->links.next) {
                            sibling = sibling->links.next;
                        }
                    }
                    sibling->links.next = &(*i);
                    i->links.previous = sibling;
                }
                i->links.parent = parent;
                sibling = &(*i);
            } else { // no parent
                i->links.initialize();
                i->links.parent = 0;
                i->links.previous = back;
                if (back) {
                    back->links.next = &(*i);
                }
                back = &(*i);
            }
        } // for loop
    }
    linkMode = newMode;
}

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

RecordData * TableImpl::addRecord(RecordId id, RecordData * parent, PTR(AuxBase) const & aux) {
    RecordSet::insert_commit_data insertData;
    if (!records.insert_check(id, CompareRecordIdLess(), insertData).second) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Record ID '%lld' is not unique.") % id).str()
        );
    }
    if (!block || block->isFull()) {
        addBlock(nRecordsPerBlock);
    }
    RecordData * p = block->makeNextRecord();
    assert(p != 0);
    p->id = id;
    p->aux = aux;
    if (linkMode == POINTERS) {
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
    } else if (linkMode == PARENT_ID) {
        p->parentId = parent->id;
    }
    records.insert_commit(*p, insertData);
    return p;
}

void TableImpl::unlink(RecordData * record) {
    if (linkMode == POINTERS && record->links.child) {
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
    if (linkMode == POINTERS) {
        if (record->links.previous) {
            record->links.previous->links.next = record->links.next;
        } else if (record->links.parent) {
            record->links.parent->links.child = record->links.next;
        }
        if (record->links.next) {
            record->links.next->links.previous = record->links.previous;
        }
        record->links.parent = 0;
    } else if (linkMode == PARENT_ID) {
        record->parentId = 0;
    }
    consolidated = 0;
}

} // namespace detail

//----- TreeIteratorBase implementation ---------------------------------------------------------------------

void TreeIteratorBase::increment() {
#ifndef NDEBUG
    if (_record._table->linkMode == PARENT_ID) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Tree iterators are invalidated when the link mode is set to PARENT_ID."
        );
    }
#endif
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

LinkMode RecordBase::getLinkMode() const { return _table->linkMode; }

Schema RecordBase::getSchema() const { return _table->schema; }

RecordBase::~RecordBase() {}

void RecordBase::setParentId(RecordId id) const {
    if (_table->linkMode == POINTERS) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot set parent ID when link mode is POINTERS."
        );
    }
    _data->parentId = id;
}

bool RecordBase::hasParent() const {
    if (_table->linkMode == PARENT_ID) {
        return _data->parentId;
    } else {
        return _data->links.parent;
    }

}

bool RecordBase::hasChildren() const {
    if (_table->linkMode == POINTERS) {
        return _data->links.child;        
    } else {
        for (detail::RecordSet::const_iterator i = _table->records.begin(); i != _table->records.end(); ++i) {
            if (i->parentId == _data->id) {
                return true;
            }
        }
        return false;
    }
}

RecordBase RecordBase::_getParent() const {
    if (_table->linkMode == POINTERS) {
        if (!_data->links.parent) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundException,
                "Record has no parent."
            );
        }
        return RecordBase(_data->links.parent, _table, *this);
    } else {
        if (!_data->parentId) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundException,
                "Record has no parent."
            );
        }
        detail::RecordSet::iterator j = _table->records.find(_data->parentId, detail::CompareRecordIdLess());
        if (j == _table->records.end()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundException,
                (boost::format("Record with id '%lld' not found.") % _data->parentId).str()
            );
        }
        return RecordBase(&(*j), _table, *this);
    }
}

TreeIteratorBase RecordBase::_beginChildren(TreeMode mode) const {
#ifndef NDEBUG
    if (_table->linkMode == PARENT_ID) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Tree iterators are invalidated when the link mode is set to PARENT_ID."
        );
    }
#endif
    return TreeIteratorBase(_data->links.child, _table, *this, mode);
}

TreeIteratorBase RecordBase::_endChildren(TreeMode mode) const {
#ifndef NDEBUG
    if (_table->linkMode == PARENT_ID) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Tree iterators are invalidated when the link mode is set to PARENT_ID."
        );
    }
#endif
    return TreeIteratorBase(
        (mode == NO_NESTING || _data->links.child == 0) ? 0 : _data->links.next,
        _table, *this, mode
    );
}

RecordBase RecordBase::_addChild(RecordId id, PTR(AuxBase) const & aux) const {
    assertBit(CAN_ADD_RECORD);
    detail::RecordData * p = _table->addRecord(id, _data, aux);
    _table->idFactory->notify(id);
    return RecordBase(p, _table, *this);
}

RecordBase RecordBase::_addChild(PTR(AuxBase) const & aux) const {
    return _addChild((*_table->idFactory)(), aux);
}

void RecordBase::unlink() const {
    assertBit(CAN_UNLINK_RECORD);
    _table->unlink(_data);
    _table->records.erase(_table->records.s_iterator_to(*_data));
}

//----- TableBase implementation --------------------------------------------------------------------------

TableBase::~TableBase() {}

LinkMode TableBase::getLinkMode() const { return _impl->linkMode; }

void TableBase::setLinkMode(LinkMode mode) const { _impl->setLinkMode(mode); }

Schema TableBase::getSchema() const { return _impl->schema; }

bool TableBase::isConsolidated() const {
    return _impl->consolidated;
}

void TableBase::consolidate(int extraCapacity, LinkMode linkMode) {
    PTR(detail::TableImpl) newImpl =
        boost::make_shared<detail::TableImpl>(
            _impl->schema,
            _impl->nRecordsPerBlock,
            _impl->idFactory->clone(),
            _impl->aux
        );
    newImpl->addBlock(_impl->records.size() + extraCapacity);
    newImpl->setLinkMode(PARENT_ID);
    int const dataOffset = sizeof(detail::RecordData);
    int const dataSize = _impl->schema.getRecordSize() - dataOffset;
    for (detail::RecordSet::const_iterator i = _impl->records.begin(); i != _impl->records.end(); ++i) {
        detail::RecordData * newRecord = newImpl->block->makeNextRecord();
        newRecord->id = i->id;
        newRecord->aux = i->aux;
        newRecord->parentId = (i->links.parent) ? i->links.parent->id : 0;
        std::memcpy(
            reinterpret_cast<char *>(newRecord) + dataOffset,
            reinterpret_cast<char const *>(&(*i)) + dataOffset,
            dataSize
        );
        newImpl->records.insert(newImpl->records.end(), *newRecord);
    }
    newImpl->setLinkMode(linkMode);
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

IteratorBase TableBase::_unlink(IteratorBase const & iter) const {
    assertBit(CAN_UNLINK_RECORD);
    _impl->assertEqual(iter._table);
    _impl->unlink(iter->_data);
    return IteratorBase(_impl->records.erase(iter.base()), _impl, iter);
}

TreeIteratorBase TableBase::_unlink(TreeIteratorBase const & iter) const {
#ifndef NDEBUG
    if (_impl->linkMode == PARENT_ID) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Tree iterators are invalidated when the link mode is set to PARENT_ID."
        );
    }
#endif
    assertBit(CAN_UNLINK_RECORD);
    _impl->assertEqual(iter->_table);
    TreeIteratorBase result(iter);
    ++result;
    _impl->unlink(iter->_data);
    _impl->records.erase(_impl->records.iterator_to(*iter->_data));
    return result;
}

TreeIteratorBase TableBase::_beginTree(TreeMode mode) const {
#ifndef NDEBUG
    if (_impl->linkMode == PARENT_ID) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Tree iterators are invalidated when the link mode is set to PARENT_ID."
        );
    }
#endif
    return TreeIteratorBase(_impl->front, _impl, *this, mode);
}

TreeIteratorBase TableBase::_endTree(TreeMode mode) const {
#ifndef NDEBUG
    if (_impl->linkMode == PARENT_ID) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Tree iterators are invalidated when the link mode is set to PARENT_ID."
        );
    }
#endif
    return TreeIteratorBase(0, _impl, *this, mode);
}

IteratorBase TableBase::_begin() const {
    return IteratorBase(_impl->records.begin(), _impl, *this);
}

IteratorBase TableBase::_end() const {
    return IteratorBase(_impl->records.end(), _impl, *this);
}

RecordBase TableBase::_get(RecordId id) const {
    detail::RecordSet::iterator j = _impl->records.find(id, detail::CompareRecordIdLess());
    if (j == _impl->records.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            (boost::format("Record with id '%lld' not found.") % id).str()
        );
    }
    return RecordBase(&(*j), _impl, *this);
}

IteratorBase TableBase::_find(RecordId id) const {
    detail::RecordSet::iterator j = _impl->records.find(id, detail::CompareRecordIdLess());
    return IteratorBase(j, _impl, *this);
}

RecordBase TableBase::_addRecord(PTR(AuxBase) const & aux) const {
    assertBit(CAN_ADD_RECORD);
    return _addRecord((*_impl->idFactory)(), aux);
}

RecordBase TableBase::_addRecord(RecordId id, PTR(AuxBase) const & aux) const {
    assertBit(CAN_ADD_RECORD);
    detail::RecordData * p = _impl->addRecord(id, 0, aux);
    _impl->idFactory->notify(id);
    return RecordBase(p, _impl, *this);
}

TableBase::TableBase(
    Schema const & schema,
    int capacity,
    int nRecordsPerBlock,
    PTR(IdFactory) const & idFactory,
    PTR(AuxBase) const & aux,
    ModificationFlags const & flags 
) : ModificationFlags(flags),
    _impl(boost::make_shared<detail::TableImpl>(schema, nRecordsPerBlock, idFactory, aux))
{
    if (capacity > 0) _impl->addBlock(capacity);
}

}}} // namespace lsst::afw::table
