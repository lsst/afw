#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

// =============== BitsColumn implementation ================================================================

namespace {

struct MatchKey {
    bool operator()(SchemaItem<Flag> const & item) const { 
        return item.key == target;
    }

    explicit MatchKey(Key<Flag> const & t) : target(t) {}

    Key<Flag> const & target;
};

struct MatchName {
    bool operator()(SchemaItem<Flag> const & item) const { 
        return item.field.getName() == target;
    }

    explicit MatchName(std::string const & t) : target(t) {}

    std::string const & target;
};

} // namespace

BitsColumn::IntT BitsColumn::getBit(Key<Flag> const & key) const {
    IntT r = std::find_if(_items.begin(), _items.end(), MatchKey(key)) - _items.begin();
    if (std::size_t(r) == _items.size()) {
        throw LSST_EXCEPT(
            pex::exceptions::NotFoundError,
            (boost::format("'%s' not found in BitsColumn") % key).str()
        );
    }
    return r;
}

BitsColumn::IntT BitsColumn::getBit(std::string const & name) const {
    IntT r = std::find_if(_items.begin(), _items.end(), MatchName(name)) - _items.begin();
    if (std::size_t(r) == _items.size()) {
        throw LSST_EXCEPT(
            pex::exceptions::NotFoundError,
            (boost::format("'%s' not found in BitsColumn") % name).str()
        );
    }
    return r;
}

BitsColumn::BitsColumn(int size) : _array(ndarray::allocate(size)) {
    _array.deep() = IntT(0);
}

// =============== BaseColumnView private Impl object =======================================================

struct BaseColumnView::Impl {
    int recordCount;                  // number of records
    void * buf;                       // pointer to the beginning of the first record's data
    PTR(BaseTable) table;             // table that owns the records
    ndarray::Manager::Ptr manager;    // manages lifetime of 'buf'

    Impl(PTR(BaseTable) const & table_, int recordCount_, void * buf_, ndarray::Manager::Ptr const & manager_)
        : recordCount(recordCount_), buf(buf_), table(table_),
          manager(manager_)
    {}
};

// =============== BaseColumnView member function implementations ===========================================

PTR(BaseTable) BaseColumnView::getTable() const { return _impl->table; }

template <typename T>
typename ndarray::ArrayRef<T,1> const BaseColumnView::operator[](Key<T> const & key) const {
    return ndarray::external(
        reinterpret_cast<T *>(
            reinterpret_cast<char *>(_impl->buf) + key.getOffset()
        ),
        ndarray::makeVector(_impl->recordCount),
        ndarray::makeVector(int(_impl->table->getSchema().getRecordSize() / sizeof(T))),
        _impl->manager
    );
}

template <typename T>
typename ndarray::ArrayRef<T,2,1> const BaseColumnView::operator[](Key< Array<T> > const & key) const {
    if (key.isVariableLength()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicError,
            "Cannot get columns for variable-length array fields"
        );
    }
    return ndarray::external(
        reinterpret_cast<T *>(
            reinterpret_cast<char *>(_impl->buf) + key.getOffset()
        ),
        ndarray::makeVector(_impl->recordCount, key.getSize()),
        ndarray::makeVector(int(_impl->table->getSchema().getRecordSize() / sizeof(T)), 1),
        _impl->manager
    );
}

ndarray::result_of::vectorize< detail::FlagExtractor, ndarray::Array< Field<Flag>::Element const,1> >::type
BaseColumnView::operator[](Key<Flag> const & key) const {
    return ndarray::vectorize(
        detail::FlagExtractor(key),
        ndarray::Array<Field<Flag>::Element const,1>(
            ndarray::external(
                reinterpret_cast<Field<Flag>::Element *>(
                    reinterpret_cast<char *>(_impl->buf) + key.getOffset()
                ),
                ndarray::makeVector(_impl->recordCount),
                ndarray::makeVector(int(_impl->table->getSchema().getRecordSize() 
                                        / sizeof(Field<Flag>::Element))),
                _impl->manager
            )
        )
    );
}

BitsColumn BaseColumnView::getBits(std::vector< Key<Flag> > const & keys) const {
    BitsColumn result(_impl->recordCount);
    ndarray::ArrayRef<BitsColumn::IntT,1,1> array = result._array.deep();
    if (keys.size() > sizeof(BitsColumn::IntT)) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            (boost::format("Too many keys passed to getBits(); %d > %d.") 
             % keys.size() % sizeof(BitsColumn::IntT)).str()
        );
    }
    BitsColumn::IntT const size = keys.size(); // just for unsigned/signed comparisons
    for (BitsColumn::IntT i = 0; i < size; ++i) {
        array |= (BitsColumn::IntT(1) << i) * (*this)[keys[i]];
        result._items.push_back(getSchema().find(keys[i]));
    }
    return result;
}

namespace {

struct ExtractFlagItems {

    template <typename T>
    void operator()(SchemaItem<T> const &) const {}

    void operator()(SchemaItem<Flag> const & item) const {
        items->push_back(item);
    }
    
    std::vector< SchemaItem<Flag> > * items;
};

} // anonymous

BitsColumn BaseColumnView::getAllBits() const {
    BitsColumn result(_impl->recordCount);
    ExtractFlagItems func = { &result._items };
    getSchema().forEach(func);
    if (result._items.size() > sizeof(BitsColumn::IntT)) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            (boost::format("Too many Flag keys in schema; %d > %d.") 
             % result._items.size() % sizeof(BitsColumn::IntT)).str()
        );
    }
    ndarray::ArrayRef<BitsColumn::IntT,1,1> array = result._array.deep();
    BitsColumn::IntT const size = result._items.size(); // just for unsigned/signed comparisons
    for (BitsColumn::IntT i = 0; i < size; ++i) {
        array |= (BitsColumn::IntT(1) << i) * (*this)[result._items[i].key];
    }
    return result;
}

// needs to be in source file so it can (implicitly) call Impl's (implicit) dtor
BaseColumnView::~BaseColumnView() {} 

BaseColumnView::BaseColumnView(
    PTR(BaseTable) const & table, int recordCount, void * buf, ndarray::Manager::Ptr const & manager
) : _impl(boost::make_shared<Impl>(table, recordCount, buf, manager)) {}

// =============== Explicit instantiations ==================================================================

#define INSTANTIATE_COLUMNVIEW_SCALAR(r, data, elem)                    \
    template ndarray::ArrayRef< elem, 1> const BaseColumnView::operator[](Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_SCALAR, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_SCALAR_FIELD_TYPE_N, AFW_TABLE_SCALAR_FIELD_TYPE_TUPLE)
)

#define INSTANTIATE_COLUMNVIEW_ARRAY(r, data, elem)                    \
    template ndarray::ArrayRef< elem, 2, 1 > const BaseColumnView::operator[](Key< Array< elem > > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_ARRAY, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_ARRAY_FIELD_TYPE_N, AFW_TABLE_ARRAY_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
