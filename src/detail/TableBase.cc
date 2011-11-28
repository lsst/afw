#include <cstring>

#include "boost/noncopyable.hpp"
#include "boost/make_shared.hpp"

#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/detail/TableBase.h"
#include "lsst/afw/table/detail/TreeIteratorBase.h"
#include "lsst/afw/table/detail/SetIteratorBase.h"
#include "lsst/afw/table/detail/Access.h"


namespace lsst { namespace afw { namespace table { namespace detail {

namespace {

class DefaultIdFactory : public IdFactory {
public:
    virtual RecordId operator()() { return ++_current; }
    DefaultIdFactory() : _current(0) {}
private:
    RecordId _current;
};

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
        double element[LayoutData::ALIGN_N_DOUBLE];
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

    void addBlock(int blockRecordCount) {
        Block::Ptr newBlock = Block::allocate(layout.getRecordSize(), blockRecordCount);
        if (block) {
            newBlock->chain.swap(block);
            consolidated = 0;
        } else {
            consolidated = newBlock->getBuffer();
        }
        block.swap(newBlock);
    }

    void assertEqual(boost::shared_ptr<TableImpl> const & other) const {
        if (other.get() != this) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                "Record and/or Iterator is not associated with this table."
            );
        }
    }

    RecordData * addRecord(RecordId id, RecordData * parent, AuxBase::Ptr const & aux) {
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
            if (parent->child) {
                RecordData * q = parent->child;
                while (q->next) {
                    q = q->next;
                }
                q->next = p;
                p->previous = q;
            } else {
                parent->child = p;
            }
            p->parent = parent;
        } else {
            if (back == 0) {
                assert(front == 0);
                front = p;
            } else {
                assert(back->next == 0);
                back->next = p;
                p->previous = back;
            }
            back = p;
        }
        records.insert_commit(*p, insertData);
        return p;
    }

    void unlink(RecordData * record) {
        if (record->child) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Children must be erased before parent record."
            );
        }
        if (!record->is_linked()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Record has already been unlinked."
            );
        }
        if (record->previous) {
            record->previous->next = record->next;
        } else if (record->parent) {
            record->parent->child = record->next;
        }
        if (record->next) {
            record->next->previous = record->previous;
        }
        consolidated = 0;
    }

    TableImpl(
        Layout const & layout_, int defaultBlockRecordCount_, 
        IdFactory::Ptr const & idFactory_, AuxBase::Ptr const & aux_
    ) :
        defaultBlockRecordCount(defaultBlockRecordCount_), front(0), back(0), consolidated(0), 
        idFactory(idFactory_), aux(aux_), layout(layout_)
    {
        Access::finishLayout(layout);
        if (!idFactory) idFactory = boost::make_shared<DefaultIdFactory>();
    }

    ~TableImpl() { records.clear(); }

};

//----- TreeIteratorBase implementation ---------------------------------------------------------------------

void TreeIteratorBase::increment() {
    switch (_mode) {
    case DEPTH_FIRST:
        if (_record._data->child) {
            _record._data = _record._data->child;
        } else if (_record._data->next) {
            _record._data = _record._data->next;
        } else {
            while (!_record._data->next && _record._data->parent) {
                _record._data = _record._data->parent;
            }
            _record._data = _record._data->next;
        }
        break;
    case NO_NESTING:
        _record._data = _record._data->next;
        break;
    }
}

//----- SetIteratorBase implementation ----------------------------------------------------------------------

SetIteratorBase::~SetIteratorBase() {}

//----- RecordBase implementation ---------------------------------------------------------------------------

Layout RecordBase::getLayout() const { return _table->layout; }

RecordBase::~RecordBase() {}

TreeIteratorBase RecordBase::_beginChildren(TreeMode mode) const {
    return TreeIteratorBase(_data->child, _table, *this, mode);
}

TreeIteratorBase RecordBase::_endChildren(TreeMode mode) const {
    return TreeIteratorBase(
        (mode == NO_NESTING || _data->child == 0) ? 0 : _data->next,
        _table, *this, mode
    );
}

RecordBase RecordBase::_addChild(RecordId id, AuxBase::Ptr const & aux) const {
    assertBit(CAN_ADD_RECORD);
    RecordData * p = _table->addRecord(id, _data, aux);
    return RecordBase(p, _table, *this);
}

RecordBase RecordBase::_addChild(AuxBase::Ptr const & aux) const {
    return _addChild((*_table->idFactory)(), aux);
}

void RecordBase::unlink() const {
    assertBit(CAN_UNLINK_RECORD);
    return _table->unlink(_data);
}

//----- TableBase implementation --------------------------------------------------------------------------

Layout TableBase::getLayout() const { return _impl->layout; }

bool TableBase::isConsolidated() const {
    return _impl->consolidated;
}

#if 0
ColumnView TableBase::consolidate() {
    if (!_impl->consolidated) {
        boost::shared_ptr<TableImpl> newStorage =
            boost::make_shared<TableImpl>(
                _impl->layout,
                _impl->defaultBlockRecordCount,
                _impl->aux
            );
        newStorage->addBlock(_impl->records.size());
        newStorage->records.reserve(_impl->records.size());
        Block & block = newStorage->blocks.back();
        int recordSize = _impl->layout.getRecordSize();
        for (
            std::vector<RecordPair>::iterator i = _impl->records.begin();
            i != _impl->records.end();
            ++i, block.next += recordSize
        ) {
            RecordPair newPair = { block.next, i->aux };
            std::memcpy(newPair.buf, i->buf, recordSize);
            newStorage->records.push_back(newPair);
        }
        _impl.swap(newStorage);
    }
    return ColumnView(
        _impl->layout, _impl->records.size(),
        _impl->consolidated, _impl->blocks.back().manager
    );
}
#endif

int TableBase::getRecordCount() const {
    return _impl->records.size();
}

SetIteratorBase TableBase::_unlink(SetIteratorBase const & iter) const {
    assertBit(CAN_UNLINK_RECORD);
    _impl->assertEqual(iter._table);
    _impl->unlink(iter->_data);
    return SetIteratorBase(_impl->records.erase(iter.base()), _impl, iter);
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

SetIteratorBase TableBase::_begin() const {
    return SetIteratorBase(_impl->records.begin(), _impl, *this);
}

SetIteratorBase TableBase::_end() const {
    return SetIteratorBase(_impl->records.end(), _impl, *this);
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

SetIteratorBase TableBase::_find(RecordId id) const {
    RecordSet::iterator j = _impl->records.find(id, detail::CompareRecordIdLess());
    return SetIteratorBase(j, _impl, *this);
}

RecordBase TableBase::_addRecord(AuxBase::Ptr const & aux) const {
    assertBit(CAN_ADD_RECORD);
    return _addRecord((*_impl->idFactory)(), aux);
}

RecordBase TableBase::_addRecord(RecordId id, AuxBase::Ptr const & aux) const {
    assertBit(CAN_ADD_RECORD);
    RecordData * p = _impl->addRecord(id, 0, aux);
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
