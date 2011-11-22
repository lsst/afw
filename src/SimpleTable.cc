#include <cstring>

#include "boost/noncopyable.hpp"
#include "boost/make_shared.hpp"

#include "lsst/afw/table/SimpleRecord.h"
#include "lsst/afw/table/SimpleTable.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

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

struct TableStorage : private boost::noncopyable {
    Layout layout;
    Block::Ptr block;
    std::vector<RecordAux::Ptr> recordAux;
    int defaultBlockRecordCount;
    int recordCount;
    RecordData * front;
    RecordData * back;
    void * consolidated;
    TableAux::Ptr aux;

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

    TableStorage(Layout const & layout_, int defaultBlockRecordCount_, TableAux::Ptr const & aux_) :
        layout(layout_), defaultBlockRecordCount(defaultBlockRecordCount_), recordCount(0), 
        front(0), back(0), consolidated(0), aux(aux_)
    {}

};

} // namespace detail

//----- SimpleRecord implementation -------------------------------------------------------------------------

Layout SimpleRecord::getLayout() const { return _storage->layout; }

SimpleRecord::~SimpleRecord() {}

//----- SimpleTable implementation --------------------------------------------------------------------------

Layout SimpleTable::getLayout() const { return _storage->layout; }

bool SimpleTable::isConsolidated() const {
    return _storage->consolidated;
}

#if 0
ColumnView SimpleTable::consolidate() {
    if (!_storage->consolidated) {
        boost::shared_ptr<detail::TableStorage> newStorage =
            boost::make_shared<detail::TableStorage>(
                _storage->layout,
                _storage->defaultBlockRecordCount,
                _storage->aux
            );
        newStorage->addBlock(_storage->records.size());
        newStorage->records.reserve(_storage->records.size());
        detail::Block & block = newStorage->blocks.back();
        int recordSize = _storage->layout.getRecordSize();
        for (
            std::vector<detail::RecordPair>::iterator i = _storage->records.begin();
            i != _storage->records.end();
            ++i, block.next += recordSize
        ) {
            detail::RecordPair newPair = { block.next, i->aux };
            std::memcpy(newPair.buf, i->buf, recordSize);
            newStorage->records.push_back(newPair);
        }
        _storage.swap(newStorage);
    }
    return ColumnView(
        _storage->layout, _storage->records.size(),
        _storage->consolidated, _storage->blocks.back().manager
    );
}
#endif

int SimpleTable::getRecordCount() const {
    return _storage->recordCount;
}

SimpleRecord SimpleTable::append(detail::RecordAux::Ptr const & aux) {
    if (!_storage->block || _storage->block->isFull()) {
        _storage->addBlock(_storage->defaultBlockRecordCount);
    }
    detail::RecordData * p = _storage->block->makeNextRecord();
    assert(p != 0);
    if (_storage->back == 0) {
        _storage->back = p;
        _storage->front = p;
    } else {
        _storage->back->sibling = p;
        _storage->back = _storage->back->sibling;
    }
    ++_storage->recordCount;
    SimpleRecord result(p, _storage);
    return result;
}

SimpleRecord SimpleTable::front() const {
    assert(_storage->front);
    return SimpleRecord(_storage->front, _storage);
}

SimpleRecord SimpleTable::back(SimpleTable::IteratorTypeEnum iterType) const {
    assert(_storage->back);
    detail::RecordData * p = _storage->back;
    if (iterType == ALL) {
        while (p->child != 0) {
            p = p->child;
        }
    }
    return SimpleRecord(p, _storage);
}

SimpleTable::SimpleTable(
    Layout const & layout,
    int defaultBlockRecordCount,
    int capacity,
    detail::TableAux::Ptr const & aux
) :
    _storage(boost::make_shared<detail::TableStorage>(layout, defaultBlockRecordCount, aux))
{
    if (capacity) _storage->addBlock(capacity);
}

SimpleTable::SimpleTable(
    Layout const & layout,
    int defaultBlockRecordCount,
    detail::TableAux::Ptr const & aux
) : _storage(boost::make_shared<detail::TableStorage>(layout, defaultBlockRecordCount, aux))
{}

SimpleTable::SimpleTable(
    Layout const & layout,
    int defaultBlockRecordCount
) : _storage(
    boost::make_shared<detail::TableStorage>(layout, defaultBlockRecordCount, detail::TableAux::Ptr())
) {}

}}} // namespace lsst::afw::table
