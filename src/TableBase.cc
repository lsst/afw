#include <cstring>

#include "boost/noncopyable.hpp"
#include "boost/make_shared.hpp"

#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/TableBase.h"
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

struct TableImpl : private boost::noncopyable {
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

    TableImpl(Layout const & layout_, int defaultBlockRecordCount_, TableAux::Ptr const & aux_) :
        layout(layout_), defaultBlockRecordCount(defaultBlockRecordCount_), recordCount(0), 
        front(0), back(0), consolidated(0), aux(aux_)
    {}

};

} // namespace detail

//----- RecordBase implementation -------------------------------------------------------------------------

Layout RecordBase::getLayout() const { return _table->layout; }

RecordBase::~RecordBase() {}

//----- TableBase implementation --------------------------------------------------------------------------

Layout TableBase::getLayout() const { return _impl->layout; }

bool TableBase::isConsolidated() const {
    return _impl->consolidated;
}

#if 0
ColumnView TableBase::consolidate() {
    if (!_impl->consolidated) {
        boost::shared_ptr<detail::TableImpl> newStorage =
            boost::make_shared<detail::TableImpl>(
                _impl->layout,
                _impl->defaultBlockRecordCount,
                _impl->aux
            );
        newStorage->addBlock(_impl->records.size());
        newStorage->records.reserve(_impl->records.size());
        detail::Block & block = newStorage->blocks.back();
        int recordSize = _impl->layout.getRecordSize();
        for (
            std::vector<detail::RecordPair>::iterator i = _impl->records.begin();
            i != _impl->records.end();
            ++i, block.next += recordSize
        ) {
            detail::RecordPair newPair = { block.next, i->aux };
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
    return _impl->recordCount;
}

RecordBase TableBase::append(detail::RecordAux::Ptr const & aux) {
    if (!_impl->block || _impl->block->isFull()) {
        _impl->addBlock(_impl->defaultBlockRecordCount);
    }
    detail::RecordData * p = _impl->block->makeNextRecord();
    assert(p != 0);
    if (_impl->back == 0) {
        _impl->back = p;
        _impl->front = p;
    } else {
        _impl->back->sibling = p;
        _impl->back = _impl->back->sibling;
    }
    ++_impl->recordCount;
    RecordBase result(p, _impl);
    return result;
}

RecordBase TableBase::front() const {
    assert(_impl->front);
    return RecordBase(_impl->front, _impl);
}

RecordBase TableBase::back(IteratorTypeEnum iterType) const {
    assert(_impl->back);
    detail::RecordData * p = _impl->back;
    if (iterType != NO_CHILDREN) {
        while (p->child != 0) {
            p = p->child;
        }
    }
    return RecordBase(p, _impl);
}

TableBase::TableBase(
    Layout const & layout,
    int defaultBlockRecordCount,
    int capacity,
    detail::TableAux::Ptr const & aux
) :
    _impl(boost::make_shared<detail::TableImpl>(layout, defaultBlockRecordCount, aux))
{
    if (capacity) _impl->addBlock(capacity);
}

}}} // namespace lsst::afw::table
