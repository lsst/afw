

#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "boost/mpl/transform.hpp"
#include "boost/fusion/algorithm/iteration/for_each.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"

#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

namespace {

struct MakeCorePair {
    template <typename T> struct apply {
        typedef boost::fusion::pair< T, ndarray::detail::Core<1>::Ptr > type;
    };
};

typedef boost::fusion::result_of::as_map<
    boost::mpl::transform< detail::ScalarFieldTypes, MakeCorePair >::type
    >::type CoreContainer;

} // anonymous

struct ColumnView::Impl {
    int recordCount;
    void * buf;
    Schema schema;
    CoreContainer cores;

    template <typename T>
    void operator()(boost::fusion::pair< T, ndarray::detail::Core<1>::Ptr > & pair) const {
        pair.second = ndarray::detail::Core<1>::create(
            ndarray::makeVector(recordCount),
            ndarray::makeVector(int(schema.getRecordSize() / sizeof(T)))
        );
    }

    Impl(Schema const & schema_, int recordCount_, void * buf_, ndarray::Manager::Ptr const & manager_)
        : recordCount(recordCount_), buf(buf_), schema(schema_)
    {
        boost::fusion::for_each(cores, *this);
    }
};

Schema ColumnView::getSchema() const { return _impl->schema; }

template <typename T>
typename ndarray::Array<T const,1> ColumnView::operator[](Key<T> const & key) const {
    return ndarray::detail::ArrayAccess< ndarray::Array<T const,1> >::construct(
        reinterpret_cast<T *>(
            reinterpret_cast<char *>(_impl->buf) + detail::Access::getOffset(key)
        ),
        boost::fusion::at_key<T>(_impl->cores)
    );
}

template <typename T>
typename ndarray::Array<T const,2,1> ColumnView::operator[](Key< Array<T> > const & key) const {
    ndarray::detail::Core<1>::Ptr scalarCore = boost::fusion::at_key<T>(_impl->cores);
    return ndarray::detail::ArrayAccess< ndarray::Array<T const,2,1> >::construct(
        reinterpret_cast<T *>(
            reinterpret_cast<char *>(_impl->buf) + detail::Access::getOffset(key)
        ),
        ndarray::detail::Core<2>::create(
            ndarray::makeVector(scalarCore->getSize(), key.getSize()),
            ndarray::makeVector(scalarCore->getStride(), 1),
            scalarCore->getManager()
        )
    );
}

ColumnView::~ColumnView() {}

ColumnView::ColumnView(
    Schema const & schema, int recordCount, void * buf, ndarray::Manager::Ptr const & manager
) : _impl(new Impl(schema, recordCount, buf, manager)) {}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_COLUMNVIEW_SCALAR(r, data, elem)                    \
    template ndarray::Array< elem const, 1> ColumnView::operator[](Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_SCALAR, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_SCALAR_FIELD_TYPE_N, AFW_TABLE_SCALAR_FIELD_TYPE_TUPLE)
)

#define INSTANTIATE_COLUMNVIEW_ARRAY(r, data, elem)                    \
    template ndarray::Array< elem const, 2, 1 > ColumnView::operator[](Key< Array< elem > > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_ARRAY, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_ARRAY_FIELD_TYPE_N, AFW_TABLE_ARRAY_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
