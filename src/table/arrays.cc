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

#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/arrays.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw { namespace table {

template <typename T>
ArrayKey<T> ArrayKey<T>::addFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc,
    std::string const & unit,
    std::vector<T> const & docData
) {
    ArrayKey<T> result;
    if (docData.empty()) return result;
    result._size = docData.size();
    result._begin = schema.addField<T>(
        schema.join(name, "0"), // we use getPrefix in order to get the version-dependent delimiter
        (boost::format(doc) % docData.front()).str(),
        unit
    );
    for (int i = 1; i < result._size; ++i) {
        schema.addField<T>(
            schema.join(name, boost::lexical_cast<std::string>(i)),
            (boost::format(doc) % docData[i]).str(),
            unit
        );
    }
    return result;
}

template <typename T>
ArrayKey<T> ArrayKey<T>::addFields(
    Schema & schema,
    std::string const & name,
    std::string const & doc,
    std::string const & unit,
    int size
) {
    ArrayKey result;
    if (size == 0) return result;
    result._size = size;
    result._begin = schema.addField<T>(
        schema.join(name, "0"), // we use getPrefix in order to get the version-dependent delimiter
        doc,
        unit
    );
    for (int i = 1; i < result._size; ++i) {
        schema.addField<T>(
            schema.join(name, boost::lexical_cast<std::string>(i)),
            doc,
            unit
        );
    }
    return result;
}

template <typename T>
ArrayKey<T>::ArrayKey(std::vector< Key<T> > const & keys) :
    _begin(),
    _size(keys.size())
{
    if (keys.empty()) return;
    _begin = keys.front();
    for (int i = 1; i < _size; ++i) {
        if (keys[i].getOffset() - _begin.getOffset() != static_cast<int>(i*sizeof(T))) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterError,
                "Keys passed to ArrayKey constructor are not contiguous"
            );
        }
    }
}

template <typename T>
ArrayKey<T>::ArrayKey(Key< Array<T> > const & other) : _begin(other[0]), _size(other.getSize()) {}

template <typename T>
ArrayKey<T>::ArrayKey(SubSchema const & s) : _begin(s["0"]), _size(1) {
    Key<T> current;
    while (true) {
        try {
            current = s[boost::lexical_cast<std::string>(_size)];
        } catch (pex::exceptions::NotFoundError &) {
            return;
        }
        if (current.getOffset() - _begin.getOffset() != static_cast<int>(_size*sizeof(T))) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterError,
                "Keys discovered in Schema are not contiguous"
            );
        }
        ++_size;
    }
}

template <typename T>
ndarray::Array<T const,1,1> ArrayKey<T>::get(BaseRecord const & record) const {
    return ndarray::external(
        record.getElement(_begin),
        ndarray::makeVector(_size),
        ndarray::ROW_MAJOR,
        record.getManager()
    );
}

template <typename T>
void ArrayKey<T>::set(BaseRecord & record, ndarray::Array<T const,1,1> const & value) const {
    LSST_THROW_IF_NE(
        value.template getSize<0>(), _size,
        pex::exceptions::LengthError,
        "Size of input array (%d) does not match size of array field (%d)"
    );
    std::copy(value.begin(), value.end(), record.getElement(_begin));
}

template <typename T>
ndarray::ArrayRef<T,1,1> ArrayKey<T>::getReference(BaseRecord & record) const {
    return ndarray::external(
        record.getElement(_begin),
        ndarray::makeVector(_size),
        ndarray::ROW_MAJOR,
        record.getManager()
    );
}

template <typename T>
ndarray::ArrayRef<T const,1,1> ArrayKey<T>::getConstReference(BaseRecord const & record) const {
    return ndarray::external(
        record.getElement(_begin),
        ndarray::makeVector(_size),
        ndarray::ROW_MAJOR,
        record.getManager()
    );
}

template <typename T>
Key<T> ArrayKey<T>::operator[](int i) const {
    if (i < 0 || i >= _size) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            "ArrayKey index does not fit within valid range"
        );
    }
    return detail::Access::makeKey<T>(_begin.getOffset() + i*sizeof(T));
}

template <typename T>
ArrayKey<T> ArrayKey<T>::slice(int begin, int end) const {
    if (begin < 0 || end > _size) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            "ArrayKey slice range does not fit within valid range"
        );
    }
    return ArrayKey((*this)[begin], end-begin);
}

template class ArrayKey<float>;
template class ArrayKey<double>;

}}} // namespace lsst::afw::table
