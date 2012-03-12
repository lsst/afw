#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

// Private implementation class for BaseColumnView
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

PTR(BaseTable) BaseColumnView::getTable() const { return _impl->table; }

template <typename T>
typename ndarray::Array<T const,1> BaseColumnView::operator[](Key<T> const & key) const {
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
typename ndarray::Array<T const,2,1> BaseColumnView::operator[](Key< Array<T> > const & key) const {
    return ndarray::external(
        reinterpret_cast<T *>(
            reinterpret_cast<char *>(_impl->buf) + key.getOffset()
        ),
        ndarray::makeVector(_impl->recordCount, key.getSize()),
        ndarray::makeVector(int(_impl->table->getSchema().getRecordSize() / sizeof(T)), 1),
        _impl->manager
    );
}

ndarray::result_of::vectorize< detail::FlagBitExtractor, ndarray::Array< Field<Flag>::Element const,1> >::type
BaseColumnView::operator[](Key<Flag> const & key) const {
    return ndarray::vectorize(
        detail::FlagBitExtractor(key),
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

BaseColumnView::~BaseColumnView() {}

BaseColumnView::BaseColumnView(
    PTR(BaseTable) const & table, int recordCount, void * buf, ndarray::Manager::Ptr const & manager
) : _impl(boost::make_shared<Impl>(table, recordCount, buf, manager)) {}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_COLUMNVIEW_SCALAR(r, data, elem)                    \
    template ndarray::Array< elem const, 1> BaseColumnView::operator[](Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_SCALAR, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_SCALAR_FIELD_TYPE_N, AFW_TABLE_SCALAR_FIELD_TYPE_TUPLE)
)

#define INSTANTIATE_COLUMNVIEW_ARRAY(r, data, elem)                    \
    template ndarray::Array< elem const, 2, 1 > BaseColumnView::operator[](Key< Array< elem > > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_ARRAY, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_ARRAY_FIELD_TYPE_N, AFW_TABLE_ARRAY_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
