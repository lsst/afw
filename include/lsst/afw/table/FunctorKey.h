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
#ifndef AFW_TABLE_FunctorKey_h_INCLUDED
#define AFW_TABLE_FunctorKey_h_INCLUDED

#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/Key.h"

namespace lsst {
namespace afw {
namespace table {

/**
 *  Base class for objects that can extract a value from a record, but are not a true Key themselves.
 *
 *  Objects that inherit from OutputFunctorKey can be passed to BaseRecord::get(), just as true Keys can,
 *  but the record will simply pass itself to OutputFunctorKey::get() and return the result.
 */
template <typename T>
class OutputFunctorKey {
public:
    virtual T get(BaseRecord const& record) const = 0;

    virtual ~OutputFunctorKey() {}
};

/**
 *  Base class for objects that can set a value on a record, but are not a true Key themselves.
 *
 *  Objects that inherit from InputFunctorKey can be passed to BaseRecord::set(), just as true Keys can,
 *  but the record will simply pass itself to OutputFunctorKey::set() along with the value that was passed.
 */
template <typename T>
class InputFunctorKey {
public:
    virtual void set(BaseRecord& record, T const& value) const = 0;

    virtual ~InputFunctorKey() {}
};

/**
 *  Convenience base class that combines the OutputFunctorKey and InputFunctorKey
 *
 *  Most objects that can set a calculated or compound value from a record can also get that value back,
 *  so we provide this class to aggregate those interfaces.
 */
template <typename T>
class FunctorKey : public OutputFunctorKey<T>, public InputFunctorKey<T> {};

/**
 *  Base class for objects that can return a non-const reference to part of a record, but are not a true Key
 *
 *  Objects that inherit from ReferenceFunctorKey can be passed to BaseRecord::operator[], just as true Keys
 *  can, but the record will simply pass itself to ReferenceFunctorKey::getReference().
 *
 *  @note We'd combine this with the ConstReferenceFunctorKey interface if it weren't for the fact that
 *  we can't pass multiple template arguments to a Swig macro if either contains commas, and we'd need that
 *  to wrap a combined interface base class.
 */
template <typename T>
class ReferenceFunctorKey {
public:
    virtual T getReference(BaseRecord& record) const = 0;

    virtual ~ReferenceFunctorKey() {}
};

/**
 *  Base class for objects that can return a const reference to part of a record, but are not a true Key
 *
 *  Objects that inherit from ConstReferenceFunctorKey can be passed to BaseRecord::operator[], just as true
 *  Keys can, but the record will simply pass itself to ReferenceFunctorKey::getConstReference().
 *
 *  @note We'd combine this with the ReferenceFunctorKey interface if it weren't for the fact that
 *  we can't pass multiple template arguments to a Swig macro if either contains commas, and we'd need that
 *  to wrap a combined interface base class.
 */
template <typename T>
class ConstReferenceFunctorKey {
public:
    virtual T getConstReference(BaseRecord const& record) const = 0;

    virtual ~ConstReferenceFunctorKey() {}
};
}
}
}  // namespace lsst::afw::table

#endif  // !AFW_TABLE_FunctorKey_h_INCLUDED
