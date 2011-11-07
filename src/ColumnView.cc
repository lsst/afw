#include "lsst/catalog/detail/fusion_limits.h"

#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "boost/mpl/transform.hpp"
#include "boost/fusion/algorithm/iteration/for_each.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"

#include "lsst/catalog/ColumnView.h"
#include "lsst/catalog/detail/KeyAccess.h"
#include "lsst/catalog/detail/FieldAccess.h"

namespace lsst { namespace catalog {

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
    char * buf;
    Layout layout;
    CoreContainer cores;

    template <typename T>
    void operator()(boost::fusion::pair< T, ndarray::detail::Core<1>::Ptr > & pair) const {
        pair.second = ndarray::detail::Core<1>::create(
            ndarray::makeVector(recordCount),
            ndarray::makeVector(int(layout.getRecordSize() / sizeof(T)))
        );
    }

    Impl(Layout const & layout_, int recordCount_, char * buf_, ndarray::Manager::Ptr const & manager_)
        : recordCount(recordCount_), buf(buf_), layout(layout_)
    {
        boost::fusion::for_each(cores, *this);
    }
};

Layout ColumnView::getLayout() const { return _impl->layout; }

template <typename T>
ColumnView::IsNullColumn ColumnView::isNull(Key<T> const & key) const {
    return ndarray::detail::ArrayAccess< ndarray::Array<int,1> >::construct(
        reinterpret_cast<int *>(
            reinterpret_cast<char *>(_impl->buf) + detail::KeyAccess::getData(key).nullOffset
        ),
        boost::fusion::at_key<int>(_impl->cores)
    ) & detail::KeyAccess::getData(key).nullMask;
}

template <typename T>
typename ndarray::Array<T const,1> ColumnView::operator[](Key<T> const & key) const {
    return ndarray::detail::ArrayAccess< ndarray::Array<T const,1> >::construct(
        reinterpret_cast<T *>(
            reinterpret_cast<char *>(_impl->buf) + detail::KeyAccess::getData(key).offset
        ),
        boost::fusion::at_key<T>(_impl->cores)
    );
}

ColumnView::~ColumnView() {}

ColumnView::ColumnView(
    Layout const & layout, int recordCount, char * buf, ndarray::Manager::Ptr const & manager
) : _impl(new Impl(layout, recordCount, buf, manager)) {}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_COLUMNVIEW_ALL(r, data, elem)                       \
    template ColumnView::IsNullColumn ColumnView::isNull(Key< elem > const &) const;

#define INSTANTIATE_COLUMNVIEW_SCALAR(r, data, elem)                    \
    template ndarray::Array< elem const, 1> ColumnView::operator[](Key< elem > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_ALL, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_SCALAR, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_SCALAR_FIELD_TYPE_N, CATALOG_SCALAR_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog
