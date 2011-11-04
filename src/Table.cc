#include "boost/noncopyable.hpp"
#include "boost/make_shared.hpp"

#include "lsst/catalog/RecordBase.h"
#include "lsst/catalog/TableBase.h"

namespace lsst { namespace catalog {

namespace detail {

namespace {

struct Block {
    char * next;
    char * end;
    ndarray::Manager::Ptr manager;

    bool isFull() const { return next == end; }

    explicit Block(int blockSize) {
        std::pair<ndarray::Manager::Ptr,char*> p = ndarray::SimpleManager<char>::allocate(blockSize);
        next = p.second;
        end = next + blockSize;
        manager = p.first;
    }
};

struct RecordPair {
    char * buf;
    Aux::Ptr aux;
};

} // anonymous

struct TableStorage : private boost::noncopyable {
    Layout layout;
    std::vector<Block> blocks;
    std::vector<RecordPair> records;
    int defaultBlockSize;
    char * consolidated;

    void addBlock(int recordCount) {
        blocks.push_back(Block(recordCount * layout.getRecordSize()));
        if (blocks.size() == 1) {
            consolidated = blocks.back().next;
        } else {
            consolidated = 0;
        }
    }

    TableStorage(Layout const & layout_, int defaultBlockSize_) :
        layout(layout_), defaultBlockSize(defaultBlockSize_), consolidated(0)
    {}

};

} // namespace detail

//----- RecordBase implementation ---------------------------------------------------------------------------

Layout RecordBase::getLayout() const { return _storage->layout; }

RecordBase::~RecordBase() {}

//----- TableBase implementation ----------------------------------------------------------------------------

Layout TableBase::getLayout() const { return _storage->layout; }

bool TableBase::isConsolidated() const {
    return _storage->consolidated;
}

ColumnView TableBase::consolidate() {
    if (!_storage->consolidated) {
        boost::shared_ptr<detail::TableStorage> newStorage =
            boost::make_shared<detail::TableStorage>(_storage->layout, _storage->defaultBlockSize);
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
            std::copy(i->buf, i->buf + recordSize, newPair.buf);
            newStorage->records.push_back(newPair);
        }
        _storage.swap(newStorage);
    }
    return ColumnView(
        _storage->layout, _storage->records.size(),
        _storage->consolidated, _storage->blocks.back().manager
    );
}

int TableBase::getRecordCount() const {
    return _storage->records.size();
}

RecordBase TableBase::operator[](int index) const {
    if (index >= static_cast<int>(_storage->records.size())) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Record index out of range."
        );
    }
    detail::RecordPair const & pair = _storage->records[index];
    return RecordBase(pair.buf, pair.aux, _storage);
}

void TableBase::erase(int index) {
    if (index >= static_cast<int>(_storage->records.size())) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Record index out of range."
        );
    }
    _storage->records.erase(_storage->records.begin() + index);
    if (index < static_cast<int>(_storage->records.size())) {
        _storage->consolidated = 0;
    }
}

RecordBase TableBase::append(Aux::Ptr const & aux) {
    if (_storage->blocks.empty() || _storage->blocks.back().isFull()) {
        _storage->addBlock(_storage->defaultBlockSize);
    }
    detail::RecordPair pair = { _storage->blocks.back().next, aux };
    _storage->records.push_back(pair);
    return RecordBase(pair.buf, aux, _storage);
}

TableBase::TableBase(Layout const & layout, int defaultBlockSize, int capacity) :
    _storage(boost::make_shared<detail::TableStorage>(layout, defaultBlockSize))
{
    if (capacity) _storage->addBlock(capacity);
}

}} // namespace lsst::catalog
