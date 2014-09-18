// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#ifndef AFW_TABLE_arrays_h_INCLUDED
#define AFW_TABLE_arrays_h_INCLUDED

#include "lsst/afw/table/FunctorKey.h"
#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief A FunctorKey used to get or set a ndarray::Array from a sequence of scalar Keys.
 */
template <typename T>
class ArrayKey : public FunctorKey< ndarray::Array<T,1,1> > {
public:

    /// Default constructor; instance will not be usuable unless subsequently assigned to.
    ArrayKey() : _begin() {}

    /// Construct from a vector of scalar Keys
    explicit ArrayKey(std::vector< Key<T> > const & keys);

    /**
     *  Construct from a compound Key< Array<T> >
     *
     *  Key< Array<T> > is now deprecated in favor of ArrayKey; this factory function is intended to
     *  aid in the transition.
     */
    template <typename T>
    explicit ArrayKey(Key< Array<T> > const & other);

    /// Return the number of elements in the array.
    int getSize() const { return _size; }

    /**
     *  @brief Construct from a subschema, assuming *_0, *_1, *_2, etc. subfields
     *
     *  If a schema has "a_0", "a_1", and "a_2" fields, this constructor allows you to construct
     *  a 3-element ArrayKey via:
     *  @code
     *  ArrayKey<T> k(schema["a"]);
     *  @endcode
     */
    ArrayKey(SubSchema const & s);

    /// Get an array from the given record
    virtual ndarray::Array<T,1,1> get(BaseRecord const & record) const;

    /// Set an array in the given record
    virtual void set(BaseRecord & record, ndarray::Array<T,1,1> const & value) const;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying x and y Keys
    bool operator==(ArrayKey<T> const & other) const {
        return other._begin == _begin && other._size == _size;
    }
    bool operator!=(ArrayKey<T> const & other) const { return !operator==(other);
    //@}

    /// Return True if both the x and y Keys are valid.
    bool isValid() const { return _begin.isValid(); }

    /// Return a scalar Key for an element of the array
    Key<T> operator[](int i) const {
        if (i < 0 || i >= size) {
            throw LSST_EXCEPT(
                pex::exceptions::LengthError,
                "ArrayKey index does not fit within valid range"
            );
        }
        return detail::Access::makeKey(_begin.getOffset() + i*sizeof(T));
    }

    /// @brief Return a FunctorKey corresponding to a range of elements
    ArrayKey slice(int begin, int end) const {
        if (begin < 0 || end > size) {
            throw LSST_EXCEPT(
                pex::exceptions::LengthError,
                "ArrayKey slice range does not fit within valid range"
            );
        }
        return ArrayKey((*this)[begin], end-begin);
    }

private:

    ArrayKey(Key<T> const & begin, int size) : _begin(begin), _size(size) {}

    Key<T> _begin;
    int _size;
};


}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_arrays_h_INCLUDED
