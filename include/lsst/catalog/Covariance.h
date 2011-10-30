// -*- c++ -*-
#ifndef CATALOG_Covariance_h_INCLUDED
#define CATALOG_Covariance_h_INCLUDED

#include "ndarray.hpp"

namespace lsst { namespace catalog {

template <typename T>
class Covariance {
public:

    // Storage is equivalent to LAPACK 'UPLO=U'
    T & operator()(int i, int j) const {
        return _flat[(i < j) ? (i + j*(j+1)/2) : (j + i*(i+1)/2)];
    } 

private:

    Covariance(ndarray::Array<T,1> const & flat) : _flat(flat) {}

    ndarray::Array<T,1> _flat;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Covariance_h_INCLUDED
