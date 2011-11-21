#include "lsst/catalog/detail/fusion_limits.h"

#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "boost/mpl/transform.hpp"
#include "boost/fusion/algorithm/iteration/for_each.hpp"
#include "boost/fusion/adapted/mpl.hpp"
#include "boost/fusion/container/map/convert.hpp"
#include "boost/fusion/sequence/intrinsic/at_key.hpp"

#include "lsst/catalog/ColumnView.h"
#include "lsst/catalog/detail/Access.h"

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
typename ndarray::Array<T,1> ColumnView::operator[](Key<T> const & key) const {
    return ndarray::detail::ArrayAccess< ndarray::Array<T,1> >::construct(
        reinterpret_cast<T *>(
            reinterpret_cast<char *>(_impl->buf) + detail::Access::getOffset(key)
        ),
        boost::fusion::at_key<T>(_impl->cores)
    );
}

template <typename T>
typename ndarray::Array<T,2,1> ColumnView::operator[](Key< Array<T> > const & key) const {
    ndarray::detail::Core<1>::Ptr scalarCore = boost::fusion::at_key<T>(_impl->cores);
    return ndarray::detail::ArrayAccess< ndarray::Array<T,2,1> >::construct(
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
    Layout const & layout, int recordCount, char * buf, ndarray::Manager::Ptr const & manager
) : _impl(new Impl(layout, recordCount, buf, manager)) {}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_COLUMNVIEW_SCALAR(r, data, elem)                    \
    template ndarray::Array< elem, 1> ColumnView::operator[](Key< elem > const &) const; \
    template ndarray::Array< elem, 2, 1 > ColumnView::operator[](Key< Array< elem > > const &) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_COLUMNVIEW_SCALAR, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_SCALAR_FIELD_TYPE_N, CATALOG_SCALAR_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog
