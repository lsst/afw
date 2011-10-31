// -*- c++ -*-
#ifndef CATALOG_Array_h_INCLUDED
#define CATALOG_Array_h_INCLUDED

#include "Eigen/Core"

#include "lsst/pex/exceptions.h"

namespace lsst { namespace catalog {

template <typename T>
class Array {
public:

    typedef Eigen::Matrix<T,Eigen::Dynamic,1> Vector;

    T & operator[](int i) const { return _data[i]; }

    int getSize() const { return _size; }

    Vector const getVector() const {
        Vector m(getSize());
        for (int i = 0; i < getSize(); ++i) {
            m[i] = _data[i];
        }
        return m;
    }

    template <typename U>
    void setVector(Eigen::MatrixBase<U> const & m) {
        if (m.size() != getSize() || !U::IsVectorAtCompileTime)
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException, "Vector has incorrect size.");
        for (int i = 0; i < getSize(); ++i) {
            _data[i] = m[i];
        }
    }

    explicit Array(T * data, int size) : _data(data), _size(size) {}

private:

    void operator=(Array const &);

    T * _data;
    int _size;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Array_h_INCLUDED
