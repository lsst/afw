// -*- c++ -*-
#ifndef CATALOG_Covariance_h_INCLUDED
#define CATALOG_Covariance_h_INCLUDED

#include "Eigen/Core"

#include "lsst/pex/exceptions.h"
#include "lsst/ndarray.h"
#include "lsst/catalog/Point.h"
#include "lsst/catalog/Shape.h"

namespace lsst { namespace catalog {

namespace detail {

// Storage is equivalent to LAPACK 'UPLO=U'
inline int indexCovariance(int i, int j) {
    return (i < j) ? (i + j*(j+1)/2) : (j + i*(i+1)/2);
}

inline int computePackedSize(int size) {
    return size * (size + 1) / 2;
}

template <typename Derived, typename T, int N>
class CovarianceColumnBase {
public:

    ndarray::ArrayRef<T,1> operator()(int i, int j) const {
        return _packed[indexCovariance(i, j)];
    }

    int getSize() const { return static_cast<Derived const &>(*this).getSize(); }

protected:

    explicit CovarianceColumnBase(
        T * buf, int recordCount, int recordStride, ndarray::Manager::Ptr const & manager, int size
    ) : _packed(
        ndarray::detail::ArrayAccess< ndarray::Array<T,2,-1> >::construct(
            buf,
            ndarray::detail::Core<2>::create(
                ndarray::makeVector(size * (size + 1) / 2, recordCount),
                ndarray::makeVector(1, recordStride),
                manager
            )
        )
    ) 
    {}

private:
    void operator=(CovarianceColumnBase const &);

    ndarray::Array<T,2,-1> _packed;
};

} // namespace detail


template <typename T>
class CovarianceColumn : public detail::CovarianceColumnBase< CovarianceColumn<T>, T, Eigen::Dynamic >
{
    typedef detail::CovarianceColumnBase< CovarianceColumn<T>, T, Eigen::Dynamic > Super;
public:

    int getSize() const { return _size; }

    CovarianceColumn(
        T * buf, int recordCount, int recordStride, 
        ndarray::Manager::Ptr const & manager, int size
    ) : Super(buf, recordCount, recordStride, manager, size), _size(size) {}

private:
    int _size;
};

template <typename U>
class CovarianceColumn< Point<U> >
    : public detail::CovarianceColumnBase< CovarianceColumn< Point<U> >, U, 2 > 
{
    typedef detail::CovarianceColumnBase< CovarianceColumn< Point<U> >, U, 2 > Super;
public:

    int getSize() const { return 2; }

    CovarianceColumn(U * buf, int recordCount, int recordStride, ndarray::Manager::Ptr const & manager) 
        : Super(buf, recordCount, recordStride, manager, 2)
    {}

};

template <typename U>
class CovarianceColumn< Shape<U> >
    : public detail::CovarianceColumnBase< CovarianceColumn< Shape<U> >, U, 3 >
{
    typedef detail::CovarianceColumnBase< CovarianceColumn< Shape<U> >, U, 3 > Super;
public:

    int getSize() const { return 3; }

    CovarianceColumn(U * buf, int recordCount, int recordStride, ndarray::Manager::Ptr const & manager)
        : Super(buf, recordCount, recordStride, manager, 3)
    {}

};

}} // namespace lsst::catalog

#endif // !CATALOG_Covariance_h_INCLUDED
