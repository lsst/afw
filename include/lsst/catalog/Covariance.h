// -*- c++ -*-
#ifndef CATALOG_Covariance_h_INCLUDED
#define CATALOG_Covariance_h_INCLUDED

namespace lsst { namespace catalog {

namespace detail {

// Storage is equivalent to LAPACK 'UPLO=U'
inline int indexCovariance(int i, int j) {
    return (i < j) ? (i + j*(j+1)/2) : (j + i*(i+1)/2);
}

inline int computeCovariancePackedSize(int size) {
    return size * (size + 1) / 2;
}

} // namespace detail

}} // namespace lsst::catalog

#endif // !CATALOG_Covariance_h_INCLUDED
