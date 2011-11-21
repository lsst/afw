#include "boost/noncopyable.hpp"
#include "boost/make_shared.hpp"

#include "lsst/afw/table/SimpleRecord.h"
#include "lsst/afw/table/SimpleTable.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

namespace {

struct AllocType {
    double element[LayoutData::ALIGN_N_DOUBLE];
};

struct Block {
    char * next;
    char * end;
    ndarray::Manager::Ptr manager;

    bool isFull() const { return next == end; }

    explicit Block(int blockSize) {
        assert(blockSize / sizeof(AllocType)); // LayoutBuilder::finish() should guarantee this.
        std::pair<ndarray::Manager::Ptr,AllocType*> p 
            = ndarray::SimpleManager<AllocType>::allocate(blockSize / sizeof(AllocType));
        next = reinterpret_cast<char*>(p.second);
        end = next + blockSize;
        manager = p.first;
    }
};

struct RecordPair {
    char * buf;
    RecordAux::Ptr aux;
};

} // anonymous

struct TableStorage : private boost::noncopyable {
    Layout layout;
    std::vector<Block> blocks;
    std::vector<RecordPair> records;
    int defaultBlockSize;
    char * consolidated;
    TableAux::Ptr aux;

    void addBlock(int recordCount) {
        blocks.push_back(Block(recordCount * layout.getRecordSize()));
        if (blocks.size() == 1) {
            consolidated = blocks.back().next;
        } else {
            consolidated = 0;
        }
    }

    TableStorage(Layout const & layout_, int defaultBlockSize_, TableAux::Ptr const & aux_) :
        layout(layout_), defaultBlockSize(defaultBlockSize_), consolidated(0), aux(aux_)
    {}

};

} // namespace detail

//----- SimpleRecord implementation ---------------------------------------------------------------------------

Layout SimpleRecord::getLayout() const { return _storage->layout; }

SimpleRecord::~SimpleRecord() {}

void SimpleRecord::initialize() const {
    
}

//----- SimpleTable implementation ----------------------------------------------------------------------------

Layout SimpleTable::getLayout() const { return _storage->layout; }

bool SimpleTable::isConsolidated() const {
    return _storage->consolidated;
}

ColumnView SimpleTable::consolidate() {
    if (!_storage->consolidated) {
        boost::shared_ptr<detail::TableStorage> newStorage =
            boost::make_shared<detail::TableStorage>(
                _storage->layout,
                _storage->defaultBlockSize,
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

int SimpleTable::getRecordCount() const {
    return _storage->records.size();
}

SimpleRecord SimpleTable::operator[](int index) const {
    if (index >= static_cast<int>(_storage->records.size())) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Record index out of range."
        );
    }
    detail::RecordPair const & pair = _storage->records[index];
    return SimpleRecord(pair.buf, pair.aux, _storage);
}

void SimpleTable::erase(int index) {
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

SimpleRecord SimpleTable::append(RecordAux::Ptr const & aux) {
    if (_storage->blocks.empty() || _storage->blocks.back().isFull()) {
        _storage->addBlock(_storage->defaultBlockSize);
    }
    detail::RecordPair pair = { _storage->blocks.back().next, aux };
    _storage->records.push_back(pair);
    _storage->blocks.back().next += _storage->layout.getRecordSize();
    SimpleRecord result(pair.buf, aux, _storage);
    return result;
}

SimpleTable::SimpleTable(
    Layout const & layout,
    int defaultBlockSize,
    int capacity,
    TableAux::Ptr const & aux
) :
    _storage(boost::make_shared<detail::TableStorage>(layout, defaultBlockSize, aux))
{
    if (capacity) _storage->addBlock(capacity);
}

SimpleTable::SimpleTable(
    Layout const & layout,
    int defaultBlockSize,
    TableAux::Ptr const & aux
) : _storage(boost::make_shared<detail::TableStorage>(layout, defaultBlockSize, aux))
{}

SimpleTable::SimpleTable(
    Layout const & layout,
    int defaultBlockSize
) : _storage(boost::make_shared<detail::TableStorage>(layout, defaultBlockSize, TableAux::Ptr()))
{}

}}} // namespace lsst::afw::table
