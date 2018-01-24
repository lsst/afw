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

namespace lsst {
namespace afw {
namespace table {

/**
 *  A FunctorKey used to get or set a ndarray::Array from a sequence of scalar Keys.
 *
 *  ArrayKey operates on the convention that arrays are defined by a set of contiguous scalar fields
 *  (i.e. added to the Schema in order, with no interruption) of the same type, with a common field
 *  name prefix and "_0", "_1" etc. suffixes.
 */
template <typename T>
class ArrayKey : public FunctorKey<ndarray::Array<T const, 1, 1> >,
                 public ReferenceFunctorKey<ndarray::ArrayRef<T, 1, 1> >,
                 public ConstReferenceFunctorKey<ndarray::ArrayRef<T const, 1, 1> > {
public:
    /**
     *  Add an array of fields to a Schema, and return an ArrayKey that points to them.
     *
     *  @param[in,out] schema  Schema to add fields to.
     *  @param[in]     name    Name prefix for all fields; "_0", "_1", etc. will be appended to this
     *                         to form the full field names.
     *  @param[in]     doc     String used as the documentation for the fields.  Should include a single
     *                         boost::format template string, which will be substituted with the
     *                         appropriate element from the docData array to form the full documentation
     *                         string.
     *  @param[in]     unit    String used as the unit for all fields.
     *  @param[in]     docData Vector of values substituted into the doc fields.  The length of the vector
     *                         determines the number of fields added.
     */
    static ArrayKey addFields(Schema& schema, std::string const& name, std::string const& doc,
                              std::string const& unit, std::vector<T> const& docData);

    /**
     *  Add an array of fields to a Schema, and return an ArrayKey that points to them.
     *
     *  @param[in,out] schema  Schema to add fields to.
     *  @param[in]     name    Name prefix for all fields; "_0", "_1", etc. will be appended to this
     *                         to form the full field names.
     *  @param[in]     doc     String used as the documentation for the fields.
     *  @param[in]     unit    String used as the unit for all fields.
     *  @param[in]     size    Number of fields to add.
     */
    static ArrayKey addFields(Schema& schema, std::string const& name, std::string const& doc,
                              std::string const& unit, int size);

    /// Default constructor; instance will not be usable unless subsequently assigned to.
    ArrayKey() : _begin() {}

    /// Construct from a vector of scalar Keys
    explicit ArrayKey(std::vector<Key<T> > const& keys);

    /**
     *  Construct from a compound Key< Array<T> >
     *
     *  Key< Array<T> > is now deprecated in favor of ArrayKey; this factory function is intended to
     *  aid in the transition.
     */
    explicit ArrayKey(Key<Array<T> > const& other);

    /**
     *  Construct from a subschema, assuming *_0, *_1, *_2, etc. subfields
     *
     *  If a schema has "a_0", "a_1", and "a_2" fields, this constructor allows you to construct
     *  a 3-element ArrayKey via:
     *
     *      ArrayKey<T> k(schema["a"]);
     */
    ArrayKey(SubSchema const& s);

    ArrayKey(ArrayKey const &);
    ArrayKey(ArrayKey &&);
    ArrayKey & operator=(ArrayKey const &);
    ArrayKey & operator=(ArrayKey &&);
    ~ArrayKey();

    /// Return the number of elements in the array.
    int getSize() const { return _size; }

    /// Get an array from the given record
    virtual ndarray::Array<T const, 1, 1> get(BaseRecord const& record) const;

    /// Set an array in the given record
    virtual void set(BaseRecord& record, ndarray::Array<T const, 1, 1> const& value) const;

    /// Get non-const reference array from the given record
    virtual ndarray::ArrayRef<T, 1, 1> getReference(BaseRecord& record) const;

    /// Get const reference array from the given record
    virtual ndarray::ArrayRef<T const, 1, 1> getConstReference(BaseRecord const& record) const;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying scalar Keys
    bool operator==(ArrayKey<T> const& other) const { return other._begin == _begin && other._size == _size; }
    bool operator!=(ArrayKey<T> const& other) const { return !operator==(other); }
    //@}

    /// Return True if the FunctorKey contains valid scalar keys.
    bool isValid() const { return _begin.isValid(); }

    /// Return a scalar Key for an element of the array
    Key<T> operator[](int i) const;

    /// Return a FunctorKey corresponding to a range of elements
    ArrayKey slice(int begin, int end) const;

private:
    ArrayKey(Key<T> const& begin, int size) : _begin(begin), _size(size) {}

    Key<T> _begin;
    int _size;
};
}
}
}  // namespace lsst::afw::table

#endif  // !AFW_TABLE_arrays_h_INCLUDED
