// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Covariance_h_INCLUDED
#define AFW_TABLE_Covariance_h_INCLUDED

namespace lsst { namespace afw { namespace table {

namespace detail {

// Storage is equivalent to LAPACK 'UPLO=U'
inline int indexCovariance(int i, int j) {
    return (i < j) ? (i + j*(j+1)/2) : (j + i*(i+1)/2);
}

inline int computeCovariancePackedSize(int size) {
    return size * (size + 1) / 2;
}

} // namespace detail

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Covariance_h_INCLUDED
